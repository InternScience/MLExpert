"""Task construction regressions: complete evidence, semantic repair and graders."""
import json
import importlib.util
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils, generator
from src.analyzer import analyze_competition, save_analysis
from src.auto_fix import auto_fix_all, auto_copy_auxiliary
from src.executor import execute_prepare, copy_description
from src.grading import check_metric, metric_semantic_errors
from src.validator import validate_all, validate_no_leakage

ROOT = Path(__file__).resolve().parents[1]
SLUG = "synthetic-regression"


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = yaml.safe_load((ROOT / "config.example.yaml").read_text())
        config["validation"]["grading_timeout"] = 10
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump(config))
        self.patches = [patch.object(utils, "PROJECT_ROOT", self.root),
                        patch.dict(os.environ, {"MLE_FACTORY_CONFIG": str(self.config)})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.raw = utils.get_raw_dir(SLUG)
        self.workspace = utils.get_workspace_dir(SLUG)
        self.private = utils.get_output_dir(SLUG) / "data/private"
        for path in (self.raw, self.workspace, self.private):
            path.mkdir(parents=True)
        self.description = "# Overview\n" + "Long official background. " * 700 + (
            "\n## Evaluation\nLATE_EVALUATION_RULE: root mean squared error, lower is better.\n"
            "## Submission\nLATE_SUBMISSION_RULE: columns id,target.\n")
        (self.raw / "description.txt").write_text(self.description)
        data = pd.DataFrame({"id": [f"row_{i:04d}" for i in range(256)],
                             "x": np.arange(256) / 10, "z": np.arange(256) % 7})
        data["target"] = 2 * data.x + 0.5 * data.z
        data.to_csv(self.raw / "train.csv", index=False)
        pd.DataFrame({"score": np.linspace(0.1, 4, 16)}).to_csv(self.private / "private_leaderboard.csv", index=False)
        pd.DataFrame({"score": np.linspace(0.1, 4, 16)}).to_csv(self.private / "public_leaderboard.csv", index=False)
        self.good_prepare = (ROOT / "examples/prepare.py").read_text()
        self.good_metric = (ROOT / "examples/metric.py").read_text()
        (self.workspace / "prepare.py").write_text(self.good_prepare)
        (self.workspace / "metric.py").write_text(self.good_metric)

    def prepare(self):
        ok, error = execute_prepare(SLUG)
        self.assertTrue(ok, error)
        copy_description(SLUG)

    def test_complete_description_reaches_analysis_generation_and_retry(self):
        analysis = analyze_competition(SLUG)
        save_analysis(SLUG, analysis)
        self.assertEqual(analysis["description"], self.description)
        # Old analysis caches can still be resumed: reload the full source text.
        analysis["description"] = self.description[:2000]
        with patch("src.generator._call_llm", return_value=self.good_prepare) as call:
            _, valid = generator.generate_prepare_script(SLUG, analysis)
            self.assertTrue(valid)
            self.assertIn("LATE_EVALUATION_RULE", call.call_args.args[0])
            self.assertIn("LATE_SUBMISSION_RULE", call.call_args.args[0])
        with patch("src.generator._call_llm", side_effect=["invalid syntax!", self.good_metric]) as call:
            _, valid = generator.generate_metric_script(SLUG, analysis)
            self.assertTrue(valid)
            self.assertEqual(call.call_count, 2)
            self.assertTrue(all("LATE_EVALUATION_RULE" in item.args[0] for item in call.call_args_list))

    def test_explicit_prompt_limit_fails_without_truncating(self):
        settings = {"api_key": "local", "base_url": "http://localhost:8000/v1",
                    "model": "local-model", "max_prompt_chars": 100}
        with patch("src.generator.get_llm_config", return_value=settings):
            with self.assertRaisesRegex(ValueError, "not truncated"):
                generator._call_llm(self.description)

    def test_postprocess_refuses_to_export_a_bad_metric(self):
        self.prepare()
        (self.workspace / "metric.py").write_text("def grade(s, a): return 0.0, False\n")
        spec = importlib.util.spec_from_file_location("factory_pipeline", ROOT / "scripts/run_pipeline.py")
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        report = pipeline.step_postprocess(SLUG)
        self.assertFalse(report["success"])
        self.assertFalse((utils.get_output_dir(SLUG) / "grade.py").exists())

    def test_validation_feedback_repairs_prepare_without_editing_metric(self):
        bad = self.good_prepare.replace(
            'answer.to_csv(private_dir / "test_answer.csv", index=False)',
            'answer.assign(target=0.0).to_csv(private_dir / "test_answer.csv", index=False)')
        (self.workspace / "prepare.py").write_text(bad)
        self.prepare()
        passed, errors = validate_all(SLUG)
        self.assertFalse(passed)
        with patch("src.generator._call_llm", return_value=self.good_prepare) as call:
            report = auto_fix_all(SLUG, errors)
        self.assertTrue(report["success"], report)
        self.assertIn("regenerate_prepare", report["attempts"][0]["actions"])
        self.assertNotIn("regenerate_metric", report["attempts"][0]["actions"])
        self.assertIn("LATE_EVALUATION_RULE", call.call_args.args[0])
        self.assertEqual((self.workspace / "metric.py").read_text(), self.good_metric)

    def test_semantically_bad_metric_triggers_metric_repair(self):
        self.prepare()
        bad = self.good_metric[:self.good_metric.index("\ndef grade(")] + (
            "\ndef grade(submission_df, answer_df):\n    score = 0.0\n    return score, False\n")
        (self.workspace / "metric.py").write_text(bad)
        passed, errors = validate_all(SLUG)
        self.assertFalse(passed)
        with patch("src.generator._call_llm", return_value=self.good_metric) as call:
            report = auto_fix_all(SLUG, errors)
        self.assertTrue(report["success"], report)
        self.assertIn("regenerate_metric", report["attempts"][0]["actions"])
        self.assertNotIn("regenerate_prepare", report["attempts"][0]["actions"])
        self.assertIn("constant", call.call_args.args[0])

    def test_arbitrary_raw_labels_are_not_automatically_copied(self):
        self.prepare()
        extracted = self.raw / "extracted"
        extracted.mkdir()
        (extracted / "hidden_labels.csv").write_text("id,label\n" + "a,1\n" * 500)
        self.assertEqual(auto_copy_auxiliary(SLUG)["copied"], [])
        self.assertFalse((utils.get_output_dir(SLUG) / "data/public/hidden_labels.csv").exists())

    def test_task_note_uses_submission_columns_not_hidden_label_columns(self):
        from src.postprocess import _build_data_note
        self.prepare()
        public = utils.get_output_dir(SLUG) / "data/public"
        pd.DataFrame({"id": ["a", "b"], "Class_A": [0.5, 0.5], "Class_B": [0.5, 0.5]}).to_csv(
            public / "sample_submission.csv", index=False)
        note = _build_data_note(SLUG)
        self.assertIn("Class_A", note)
        self.assertIn("Class_B", note)
        self.assertNotIn("`target`", note)

    def test_original_sample_submission_filename_survives_export(self):
        from src.postprocess import fix_description, generate_quality_report
        from src.exporter import export_task
        self.prepare()
        public = utils.get_output_dir(SLUG) / "data/public"
        (public / "sample_submission.csv").rename(public / "sampleSubmission.csv")
        self.assertTrue(validate_all(SLUG)[0])
        fix_description(SLUG)
        report = generate_quality_report(SLUG)
        self.assertEqual(report["status"], "pass", report)
        (self.workspace / "quality_report.json").write_text(json.dumps(report))
        export_task(SLUG)
        self.assertIn("sampleSubmission.csv", (utils.get_output_dir(SLUG) / "README.md").read_text())
        self.assertIn("sampleSubmission.csv", (public / "description.txt").read_text())

    def test_small_and_zero_based_ids_are_checked_for_leakage(self):
        self.prepare()
        public = utils.get_output_dir(SLUG) / "data/public"
        pd.DataFrame({"id": range(3), "target": [0, 1, 2]}).to_csv(self.private / "test_answer.csv", index=False)
        pd.DataFrame({"id": range(3), "x": [1, 2, 3], "target": [0, 1, 2]}).to_csv(public / "train.csv", index=False)
        self.assertTrue(any("LEAKAGE" in error for error in validate_no_leakage(SLUG)))


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "metric.py"
        self.good = (ROOT / "examples/metric.py").read_text()
        self.answer = pd.DataFrame({"id": [f"id_{i}" for i in range(30)], "target": np.arange(30) / 3})
        self.submission = self.answer.assign(target=0.0)

    def check(self, code, timeout=10):
        self.path.write_text(code)
        return check_metric(self.path, self.submission, self.answer, timeout=timeout)

    def test_valid_regression_metric(self):
        report = self.check(self.good)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["response_checked"])

    def test_numeric_ids_and_integer_dummy_predictions(self):
        self.path.write_text(self.good)
        answer = self.answer.assign(id=np.arange(len(self.answer)))
        submission = answer.assign(target=0)
        report = check_metric(self.path, submission, answer)
        self.assertEqual(report["errors"], [])

    def test_constant_and_nonfinite_scores_are_rejected(self):
        for value in ("0.0", 'float("inf")', 'float("nan")'):
            with self.subTest(value=value):
                code = self.good[:self.good.index("\ndef grade(")] + (
                    f"\ndef grade(submission_df, answer_df):\n    score = {value}\n    return score, False\n")
                self.assertTrue(self.check(code)["errors"])

    def test_wrong_metric_direction_is_rejected(self):
        self.assertTrue(self.check(self.good.replace("return score, False", "return score, True"))["errors"])

    def test_row_order_bug_is_detected_with_constant_dummy_predictions(self):
        bad = self.good.replace("predictions.reindex(answers.index) - answers",
                                "pd.Series(predictions.to_numpy() - answers.to_numpy())")
        report = self.check("import pandas as pd\n" + bad)
        self.assertTrue(any("row order" in error for error in report["errors"]), report)

    def test_invalid_submission_acceptance_is_rejected(self):
        code = "def validate_submission(submission_df, answer_df):\n    return 'Valid'\n" + self.good[self.good.index("\ndef grade("):]
        self.assertTrue(self.check(code)["errors"])

    def test_metric_hang_is_stopped(self):
        self.assertTrue(any("timed out" in error for error in self.check("while True: pass", timeout=0.5)["errors"]))

    def test_plain_log_rmse_rejects_log1p_substitution(self):
        description = ("## Evaluation\nRoot-Mean-Squared-Error (RMSE) between the logarithm "
                       "of the predicted value and the logarithm of the observed value.\n")
        self.path.write_text("from sklearn.metrics import mean_squared_log_error as msle\ndef grade(s,a): return msle(s,a),False\n")
        self.assertTrue(metric_semantic_errors(self.path, description))
        self.assertEqual(metric_semantic_errors(self.path, "## Evaluation\nRMSLE using log1p.\n"), [])
        self.path.write_text("import numpy as np\ndef grade(s,a): return np.mean((np.log(s)-np.log(a))**2)**0.5,False\n")
        self.assertEqual(metric_semantic_errors(self.path, description), [])

    def test_probability_submission_and_label_answer_can_differ(self):
        code = '''import numpy as np
from sklearn.metrics import log_loss
def validate_submission(submission_df, answer_df):
    if set(submission_df.columns) != {"id", "a", "b", "c"}:
        raise ValueError("wrong columns")
    if len(submission_df) != len(answer_df) or set(submission_df.id) != set(answer_df.id):
        raise ValueError("wrong IDs")
    if submission_df.id.duplicated().any():
        raise ValueError("duplicate IDs")
def grade(submission_df, answer_df):
    validate_submission(submission_df, answer_df)
    ordered = submission_df.set_index("id").reindex(answer_df.id)
    return float(log_loss(answer_df.label, ordered[["a", "b", "c"]], labels=["a", "b", "c"])), False
'''
        self.path.write_text(code)
        answer = pd.DataFrame({"id": [f"id_{i}" for i in range(30)], "label": ["a", "b", "c"] * 10})
        submission = answer[["id"]].assign(a=1/3, b=1/3, c=1/3)
        report = check_metric(self.path, submission, answer)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["response_checked"])


class PipelineReportingTests(unittest.TestCase):
    def test_temperature_defaults_to_one_and_accepts_configuration(self):
        for configured, expected in (({}, 1.0), ({"temperature": 0.7}, 0.7)):
            with self.subTest(configured=configured):
                client = MagicMock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))])
                settings = {"model": "local-model", "base_url": "http://localhost:8000/v1",
                            "api_key": "local", **configured}
                with patch("src.generator.get_llm_config", return_value=settings), patch.dict(
                    sys.modules, {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=client))}
                ):
                    self.assertEqual(generator._call_llm("Generate a task"), "generated code")
                self.assertEqual(client.chat.completions.create.call_args.kwargs["temperature"], expected)

    def test_download_success_is_not_reported_as_a_ready_task(self):
        spec = importlib.util.spec_from_file_location("factory_report", ROOT / "scripts/run_pipeline.py")
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        output = io.StringIO()
        with redirect_stdout(output):
            pipeline.print_report([{"slug": "example", "status": "success", "total_time": 1,
                                    "steps": {"download": {"success": True}}}])
        self.assertNotIn("Ready to use", output.getvalue())


if __name__ == "__main__":
    unittest.main()
