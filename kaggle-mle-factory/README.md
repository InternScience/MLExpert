# Kaggle-MLE Factory

Convert a Kaggle competition into an executable MLE task: download its assets, retrieve the official description, generate preparation and scoring code, validate the outputs, and export a task package. Execution and validation failures receive bounded automatic repair attempts.

The pipeline accompanies [MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering](../paper/MLExpert_agentic_mle_arxiv.pdf).

## Quick start

Use Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp config.example.yaml config.yaml
```

Install Chromium system dependencies if required by your host. Set your model service in `config.yaml`:

```yaml
llm:
  model: "your-model"
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"
  temperature: 1.0
```

The service must support Chat Completions with `temperature` and `max_tokens`. Model name, endpoint, and API key can also be set through `MLE_FACTORY_LLM_MODEL`, `MLE_FACTORY_LLM_BASE_URL`, and `MLE_FACTORY_LLM_API_KEY`.

Configure your Kaggle account using `KAGGLE_USERNAME` and `KAGGLE_KEY`, or fill in the credential template:

```bash
mkdir -p ~/.kaggle
cp kaggle.example.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Accept the competition's rules and confirm download access, then build its task:

```bash
python scripts/run_pipeline.py --slug competition-slug
```

## Usage

Install the dependencies in Quick start before running the offline example or pipeline commands.

```bash
# Build explicitly selected competitions.
python scripts/run_pipeline.py --slugs first-competition second-competition

# Run one stage or resume a previous build.
python scripts/run_pipeline.py --slug competition-slug --step analyze
python scripts/run_pipeline.py --slug competition-slug --from prepare

# Use a local raw/competition-slug/description.txt and skip webpage retrieval.
python scripts/run_pipeline.py --slug competition-slug --skip-scrape

# Run the synthetic example without credentials or a model service.
python examples/offline_demo.py
```

When using `--skip-scrape`, supply a description covering the task, evaluation metric, submission format, and data details.

Stages run in order: `download`, `scrape`, `analyze`, `generate`, `prepare`, `validate`, `postprocess`. Failed tasks stop with a nonzero exit status. Completed builds include a standalone grader:

```bash
python output/competition-slug/grade.py /path/to/submission.csv
```

Competition discovery is optional: `python -m src.discovery` writes candidates to `data/candidates.json`. Add discovery exclusions using `data/exclusion_list.example.txt`. Explicit `--slug` and `--slugs` inputs bypass selection.

## Configuration

`config.example.yaml` lists the available settings. Local `config.yaml` and runtime artifacts are ignored by Git.

| Setting | Purpose |
| --- | --- |
| `llm.temperature` | Sampling temperature; default `1.0`. |
| `llm.max_tokens` | Initial generation budget; default `8192`. Regeneration uses `16384`. |
| `llm.timeout`, `llm.max_retries` | Model request timeout and transport retries. |
| `llm.max_prompt_chars` | Optional input-size limit; `0` sends the complete evidence. Exceeding a limit raises an error. |
| `prepare.timeout` | Preparation subprocess timeout. |
| `prepare.auxiliary_files` | Explicit feature files allowed for automatic copying. |
| `validation.grading_timeout` | Scoring subprocess timeout. |
| `validation.max_grading_rows` | Maximum sampled submission rows, aligned with answers by ID; `0` checks all rows. |
| `validation.repair_rounds` | Maximum validation-driven repair rounds. |

Set `MLE_FACTORY_ROOT` to keep runtime data outside the source tree, and `MLE_FACTORY_CONFIG` to select another configuration file. Standard proxy environment variables and `KAGGLE_CONFIG_DIR` are supported.

Generation and repair receive complete task descriptions, sampled data profiles, and installed package versions. Run generated Python in a container or another isolated construction environment. The configured model service receives the task evidence.

## Task package

```text
output/<slug>/
  README.md, requirements.txt
  prepare.py, metric.py, grade.py
  quality_report.json
  data/
    public/     # Description, training data, test inputs, submission example
    private/    # Held-out answers and available leaderboard scores
```

Give the solving agent access to `data/public/` and keep evaluator files separate. Original filenames, including `sampleSubmission.csv`, are preserved and documented in the exported README.

Preparation scripts implement `prepare(raw_dir, public_dir, private_dir)`. Metrics implement `validate_submission(submission_df, answer_df)` and `grade(submission_df, answer_df)`, returning `(score, higher_is_better)`. The grader prints these values as JSON.

Validation checks data structure, selected leakage patterns, score finiteness, invalid submissions, row-order invariance, explicit log-transform mismatches, and prediction sensitivity where an answer-derived probe is available.

Intermediate scripts and diagnostics are stored in `workspace/<slug>/`; validation-triggered repairs produce `repair_report.json`.

## Important notes

**Human review.** Check generated data splits and scoring code against the competition rules before training or evaluation. Models may simplify complex metrics or omit scoring steps. Automated checks are recorded in `quality_report.json`; original leaderboards provide an approximate reference for reconstructed splits.

**Kaggle access and data use.** Use your own Kaggle account, accept the competition rules, and follow the applicable data and content licenses.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

Code is licensed under [Apache 2.0](LICENSE). Competition data and descriptions retain their source terms and are downloaded separately.
