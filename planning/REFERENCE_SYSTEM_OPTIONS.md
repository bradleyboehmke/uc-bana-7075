# Reference System Options

**Recommendation:** Option A, a simplified course template derived from Rossmann `main`, confirmed by the instructor during planning (decision D-07 in [DECISION_LOG.md](DECISION_LOG.md)) and recorded in [ADR 0001](../docs/adr/0001-rossmann-as-reference-system.md). It has the highest weighted total (76 of 95) and scores highest or tied on five of eight criteria. Its main risk, Kaggle data redistribution, was cleared on 2026-11-19: the rules permit reuse (D-37).

Evidence for each score is in [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md).

## Options

| Option | Description |
|---|---|
| **A. Simplified Rossmann** | New template repository derived from [rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting) `main`: single LightGBM model, Great Expectations 1.x, down-sampled stores, committed sample data, trimmed features, SHAP in code only, devcontainer and Colab notebooks added |
| **B. Modernized apple-demand stack** | Extend the course's [ModelOps/](../ModelOps/) code into a full system: add a data pipeline, validation, tests, and CI; YouTube pipeline stays for DataOps or is retired |
| **C. New retail demand system** | Build a new forecasting system from scratch on a public, redistributable retail dataset |
| **D. Other new business application** | Build a new system for a different problem type, such as churn classification |

## Decision matrix

Scores run 1 (poor) to 5 (strong). Weights reflect the redesign's constraints: accessibility and maintenance matter most because the course launches in about 14 weeks and serves non-coders.

| Criterion | Weight | A. Rossmann | B. Apple stack | C. New retail | D. Other new |
|---|---|---|---|---|---|
| Business relevance | 2 | 5: real retailer, 1,115 stores, promotions, holidays, competition | 3: synthetic apple demand, plausible but invented | 4: depends on dataset | 4: depends on dataset |
| Instructional coverage of spec §6 | 3 | 5: validation, tracking, registry, API, app, monitoring, tests, CI, retraining all exist | 2: no pipeline, validation, tests, or CI | 1: everything to build | 1: everything to build |
| Technical complexity (5 = right-sized) | 2 | 3: about 12,000 lines and a three-model ensemble; must be cut down | 4: small and readable | 4: can be designed to size | 4: can be designed to size |
| Student accessibility | 3 | 4 after simplification: Streamlit UI and MLflow UI give non-coders a way in | 3: notebooks only, apart from a small Streamlit app | 4: can be designed for it | 4: can be designed for it |
| Reusability of existing assets | 2 | 5: maturity branches for Week 1, existing docs and screenshots, tests | 4: notebooks referenced throughout chapters 06–09 | 1 | 1 |
| Hosting requirements | 2 | 3: needs Codespaces for MLflow, API, and app; data must be shrunk | 4: small enough for Colab | 4 | 4 |
| Maintenance cost | 3 | 3: large codebase, unpinned dependencies | 3: legacy Evidently API, corporate Dockerfile | 2: new code with no history | 2: new code with no history |
| Controlled failure scenarios | 2 | 4: drift batches exist; other failures easy to add to an existing test suite | 5: drift generator built in | 3: must be designed | 3: must be designed |
| **Weighted total (max 95)** | | **76** | **64** | **53** | **53** |

## Why A over B

B is smaller and has a built-in drift generator, but it is missing most of what the course teaches: a data pipeline, validation, automated tests, and CI. Bringing B up to A's coverage means writing those parts new, which is the work C and D require anyway. A starts with them and needs subtraction instead.

A's maturity branches are unique: they let Week 1 show the same project as notebooks, as modular code, and as a production system, which ties directly to the Hidden Technical Debt reading.

B does not disappear. Its small notebooks become Build It labs and a possible Week 3 supplementary example, and `apple_data.py` is the model for building controlled drift into the template.

## Risks of A and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle rules might not permit redistributing `train.csv` | **Resolved:** the instructor confirmed the rules permit reuse (D-37); the template ships down-sampled real data |
| Simplification takes longer than expected | Simplify by deletion first (ensemble, Optuna, extra features), modernize second (Great Expectations 1.x). Keep a working build at each step |
| Upstream Rossmann changes break the template | The template is a separate repository with pinned dependencies and no automatic sync |
| Codespaces quota | 120 compute hours a month on GitHub Free; Try Its need well under 10 |
| Large binaries slow Codespace starts | Template excludes the 49 MB committed models and ships only down-sampled data |

## What the simplified template must contain

- One business problem: daily sales forecasting for a down-sampled set of stores, sized so the full pipeline runs in a few minutes in Colab.
- Pipeline stages: ingest, validate raw data, build features, validate features, train, evaluate, register, serve, log predictions, monitor.
- A single LightGBM model with fixed hyperparameters; SHAP values available in code.
- Great Expectations 1.x suites with browsable Data Docs.
- MLflow tracking and registry with a `champion` alias.
- FastAPI service and Streamlit app, including the Monitoring page.
- Evidently drift and performance reports.
- pytest suite and a GitHub Actions workflow.
- Packaged failure scenarios: an invalid-data batch, a breaking change that fails a test, a drifted batch, a performance-degradation batch, and a documented architectural flaw for teardown.
- Precomputed artifacts at each stage so students can start any week without completing the previous one.
- A devcontainer, pinned dependencies with a lockfile, and one Colab notebook per notebook-based Try It.
