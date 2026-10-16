"""Regressions for portable construction and truthful pipeline results."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils
from src.executor import _run_prepare
from src.scraper import _get_proxy

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_config_environment_overrides_without_private_file(self):
        env = {
            "MLE_FACTORY_CONFIG": str(ROOT / "config.example.yaml"),
            "MLE_FACTORY_LLM_MODEL": "local-model",
            "MLE_FACTORY_LLM_BASE_URL": "http://localhost:8000/v1",
            "MLE_FACTORY_LLM_API_KEY": "local",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = utils.load_config()
            self.assertEqual(utils.get_llm_config()["model"], "local-model")
            self.assertEqual(cfg["paths"]["output_dir"], "output")

    def test_no_implicit_proxy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(utils.get_proxy_url({}), "")
        with patch("src.scraper.get_proxy_url", return_value=""):
            self.assertIsNone(_get_proxy())

    def test_kaggle_environment_credentials(self):
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "example", "KAGGLE_KEY": "test-key"}, clear=True):
            self.assertEqual(utils.load_kaggle_credentials(), {"username": "example", "key": "test-key"})

    def test_slug_cannot_escape_data_root(self):
        for slug in ("../data", "/tmp/data", "..", "a/b", "a\\b"):
            with self.subTest(slug=slug), self.assertRaises(ValueError):
                utils.get_output_dir(slug)

    def test_prepare_marker_does_not_override_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "a 'quoted' directory"
            root.mkdir()
            prepare = root / "prepare.py"
            prepare.write_text('print("PREPARE_SUCCESS")\nraise RuntimeError("failed")\n', encoding="utf-8")
            ok, message = _run_prepare("synthetic", root, prepare, root, root, root, 30)
            self.assertFalse(ok)
            self.assertIn("RuntimeError", message)

    def test_failed_cli_has_nonzero_exit_and_failure_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, MLE_FACTORY_ROOT=temporary,
                       MLE_FACTORY_CONFIG=str(ROOT / "config.example.yaml"))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_pipeline.py"),
                 "--slug", "missing-data", "--step", "analyze"],
                env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            log = json.loads((Path(temporary) / "data/processing_log.json").read_text())
            self.assertEqual(log[0]["status"], "failed")

    def test_offline_demo_without_network_or_model_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Propagate a network ban into every Python subprocess in the demo.
            (root / "sitecustomize.py").write_text(
                "import socket\n"
                "def blocked(*args, **kwargs):\n"
                "    raise RuntimeError('Network access forbidden in offline demo')\n"
                "socket.socket.connect = blocked\n"
                "socket.socket.connect_ex = blocked\n"
                "socket.create_connection = blocked\n"
                "socket.getaddrinfo = blocked\n", encoding="utf-8",
            )
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MLE_FACTORY_", "KAGGLE_"))}
            env["PYTHONPATH"] = str(root)
            destination = root / "demo with 'quotes'"
            result = subprocess.run(
                [sys.executable, str(ROOT / "examples/offline_demo.py"),
                 "--work-dir", str(destination)],
                env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            log = json.loads((destination / "data/processing_log.json").read_text())
            self.assertEqual(log[0]["status"], "success")
            self.assertEqual(set(log[0]["steps"]), {"prepare", "validate", "postprocess"})
            report = json.loads((destination / "workspace/synthetic-regression/quality_report.json").read_text())
            self.assertEqual(report["status"], "pass")
            self.assertTrue((destination / "submission.csv").is_file())
            # A user should be able to copy the task elsewhere and grade a
            # submission without importing the Factory's src package.
            relocated = root / "standalone task"
            shutil.copytree(destination / "output/synthetic-regression", relocated)
            graded = subprocess.run(
                [sys.executable, str(relocated / "grade.py"), str(destination / "submission.csv")],
                cwd=root, env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(graded.returncode, 0, graded.stdout + graded.stderr)
            self.assertLess(json.loads(graded.stdout)["score"], 1e-10)
            bad_submission = root / "invalid.csv"
            bad_submission.write_text("id,target\nmissing-id,0\n", encoding="utf-8")
            rejected = subprocess.run(
                [sys.executable, str(relocated / "grade.py"), str(bad_submission)],
                cwd=root, env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)


if __name__ == "__main__":
    unittest.main()
