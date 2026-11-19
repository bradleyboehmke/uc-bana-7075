# Current State Audit

**Audited:** 2026-11-19, `uc-bana-7075` at commit `2612731` and `rossmann-forecasting` `main` at its 2026-01-23 commit ("Update docs").
**Scope:** everything tracked in both repositories, plus untracked local folders that affect planning. Assignments and quizzes live in Canvas and were not audited.

## 1. Course repository architecture

| Area | Path | What it is |
|---|---|---|
| Textbook | [book/](../book/) | Quarto book, 13 chapters + 1 appendix, 39 images, published to GitHub Pages |
| Slides | [slides/](../slides/) | 8 Quarto reveal.js decks covering weeks 1–5 only |
| DataOps code | [DataOps/](../DataOps/) | YouTube data pipeline notebook and helpers |
| ModelOps code | [ModelOps/](../ModelOps/) | Apple-demand experiments, registry, FastAPI/Streamlit serving, Evidently monitoring |
| Readings | [papers/](../papers/) | Hidden Technical Debt (NeurIPS 2015) and Booking.com "150 Successful ML Models" (`bernardi2019.pdf`) |
| Publishing | [.github/workflows/publish-book.yml](../.github/workflows/publish-book.yml) | Renders `book/` with Quarto **pre-release** and publishes to `gh-pages`; runs on push, PR, manual, and daily at 23:00 |
| Environment | [bana7075.yml](../bana7075.yml) | Conda env with only `python>=3.10`, `jupyter`, `nbformat` |
| Untracked, gitignored | `grading/`, `*Syllabus*`, `recordings/` | Student submissions and grading notes, syllabi, lecture recordings. Not public; must stay out of `planning/` |

The repository is public (the book is published from it), so planning documents must not contain student names or grades.

### Textbook

| Chapter | Lines | Main content | Has exercise | Has case studies |
|---|---|---|---|---|
| [01-intro-ml-system.qmd](../book/01-intro-ml-system.qmd) | 256 | ML system design, design principles, lifecycle | Yes | Yes (Netflix, Uber, Spotify, Airbnb, Google TFX) |
| [02-before-we-build.qmd](../book/02-before-we-build.qmd) | 465 | Stakeholders, deciding whether ML is needed, requirements | Yes | No |
| [03-dataops-role.qmd](../book/03-dataops-role.qmd) | 1054 | Ingestion, processing, sampling, missing data, validation, versioning | Yes | No |
| [04-dataops-build.qmd](../book/04-dataops-build.qmd) + `.ipynb` | 649 | ETL vs ELT, DataOps tools, YouTube pipeline walkthrough | Yes | No |
| [05-modelops-role.qmd](../book/05-modelops-role.qmd) | 261 | ModelOps components | Yes | Yes |
| [06-modelops-experimenting.qmd](../book/06-modelops-experimenting.qmd) | 478 | Experiment tracking, MLflow | Yes | No |
| [07-modelops-versioning.qmd](../book/07-modelops-versioning.qmd) | 534 | Model versioning and registry | Yes | No |
| [08-modelops-deployment.qmd](../book/08-modelops-deployment.qmd) | 1053 | Deployment considerations and strategies, FastAPI | Yes | No |
| [09-modelops-monitoring.qmd](../book/09-modelops-monitoring.qmd) | 864 | Performance, data and concept drift, Evidently | Yes | No |
| [10-devops-role.qmd](../book/10-devops-role.qmd) | 329 | DevOps for ML, CI/CD concepts | Yes (Hidden Technical Debt paper) | No |
| [11-devops-git.qmd](../book/11-devops-git.qmd) | 801 | Git, GitHub, branching, PRs, GitHub Actions; command-line based | Yes | No |
| [12-human-side-of-ml.qmd](../book/12-human-side-of-ml.qmd) | 1027 | Collaboration, human-in-the-loop, organizational change, responsible AI | Yes | Yes |
| [13-continued-learning.qmd](../book/13-continued-learning.qmd) | 77 | Career growth | No | No |
| [99-ml-serving-methods.qmd](../book/99-ml-serving-methods.qmd) | 139 | Serving method selection criteria | No | No |

Callouts are used heavily (12 to 79 per chapter), and [_quarto.yml](../book/_quarto.yml) already sets `callout-appearance: simple`, so the Human Side and Build with AI callouts fit the existing conventions. No chapter mentions Colab.

### Slides

[00-intro](../slides/00-intro.qmd), [01-ml-system-design](../slides/01-ml-system-design.qmd), [02-before-we-build](../slides/02-before-we-build.qmd) and its [in-class exercise](../slides/02-before-we-build-exercise.qmd), [03-dataops](../slides/03-dataops.qmd) ("Week 2"), and [04](../slides/04-modelops.qmd), [05](../slides/05-modelops.qmd), [06-modelops](../slides/06-modelops.qmd) ("Weeks 3–5"). There are no slides for DevOps, monitoring as its own week, the human side, or the final week.

### DataOps code

- [youtube-data-pipeline.ipynb](../DataOps/youtube-data-pipeline.ipynb) with [dataops_utils.py](../DataOps/dataops_utils.py): ingests YouTube video stats and transcripts, validates with Great Expectations, versions with DVC.
- Requires a `YOUTUBE_API_KEY` and calls `youtube_transcript_api` ([dataops_utils.py:5](../DataOps/dataops_utils.py#L5)), which cloud IPs such as Colab's are often blocked from.
- Data is DVC-tracked ([youtube_video_data.parquet.dvc](../DataOps/data/youtube_video_data.parquet.dvc)) and no remote is configured, so a fresh clone has no data.
- Pins `great_expectations==1.3.1` ([dataops-requirements.txt](../DataOps/dataops-requirements.txt)): a working Great Expectations 1.x example already exists in the course.

### ModelOps code

All of it uses synthetic apple demand data from [apple_data.py](../ModelOps/apple_data.py), whose generator takes `feature_drift_factors` and `concept_drift_factors`.

| File | Purpose |
|---|---|
| [model-experimentation.ipynb](../ModelOps/model-experimentation.ipynb) | MLflow experiment tracking |
| [model-registration.ipynb](../ModelOps/model-registration.ipynb), [model-retrieval.ipynb](../ModelOps/model-retrieval.ipynb) | Registry and aliases (`apple_demand@champion`) |
| [fastapi_app.py](../ModelOps/fastapi_app.py), [streamlit_app.py](../ModelOps/streamlit_app.py), [fastapi_streamlit.py](../ModelOps/fastapi_streamlit.py) | Serving |
| [fastapi_with_monitoring_app.py](../ModelOps/fastapi_with_monitoring_app.py), [fastapi_streamlit_with_monitoring.py](../ModelOps/fastapi_streamlit_with_monitoring.py), [clean_prediction_logs.py](../ModelOps/clean_prediction_logs.py) | Prediction logging |
| [model-drift-monitoring.ipynb](../ModelOps/model-drift-monitoring.ipynb) / [.py](../ModelOps/model-drift-monitoring.py) and the three `*_report.html` files | Evidently drift and performance reports |
| [model-deploy/](../ModelOps/model-deploy/) | Dockerfile and docker-compose |
| `mlruns/` (257 tracked files, 4.7 MB), `training_data/` | Committed MLflow store and training snapshots |

Book chapters 06–09 reference `ModelOps/` paths 47 times (06: 12, 07: 6, 08: 19, 09: 10).

## 2. Rossmann repository

[rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting) shows one forecasting problem at three maturity levels on branches `1-initial-exploration`, `2-research-mature`, and `main`.

| Area | Evidence on `main` |
|---|---|
| Data | `data/raw/train.csv` (1,017,210 rows, 36 MB with `store.csv`) committed; processed parquet files DVC-tracked to a **local** remote (`.dvc/config`: `url = .dvc/cache`) |
| Pipeline | [dvc.yaml](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/dvc.yaml): prepare → validate raw → build features → validate features; training stages commented out |
| Validation | Great Expectations suites for raw, processed, and feature data, pinned `<1.0`; two config directories (`great_expectations/` and `gx/`) |
| Features | `config/params.yaml`: lags 1/7/14/28, rolling windows 7/14/28/60, promo and competition features |
| Models | XGBoost + LightGBM + CatBoost ensemble with Optuna tuning; 49 MB of final models committed under `models/final/` |
| Tracking and registry | MLflow ([mlflow_utils.py](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/src/utils/mlflow_utils.py), [model_registry.py](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/src/models/model_registry.py), [validate_model.py](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/src/models/validate_model.py)) |
| Serving | FastAPI ([deployment/api/main.py](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/deployment/api/main.py)) with a prediction logger; multipage Streamlit app (Predictions, Documentation, Monitoring) |
| Monitoring | Evidently (new API: `DataDefinition`, `Dataset`, `presets`), reference data preparation, batch data for Aug–Dec 2015 |
| Tests and CI | 10 pytest files, 159 test functions; [test.yml](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/.github/workflows/test.yml) runs them on push and PR; [docs.yml](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/.github/workflows/docs.yml) builds an MkDocs site |
| Operations | Shell launchers for MLflow, API, Streamlit, retraining, and the DataOps workflow; pre-commit config |
| Size | About 12,000 lines of Python across `src/`, `deployment/`, and `tests/` |
| Missing | Explainability (no SHAP), devcontainer, lockfile, Colab notebooks |

## 3. External checks

| Question | Finding |
|---|---|
| Codespaces free tier | 120 compute hours and 15 GB-month storage per month for GitHub Free personal accounts ([GitHub docs](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces), 2026-11-19). No separate student allowance is mentioned on that page |
| Kaggle Rossmann redistribution | **Permitted**, per the instructor's check of the competition rules on 2026-11-19 (D-37). The rules page could not be loaded during the automated audit |
| Great Expectations 1.x migration | Feasible. Rossmann's [validate_data.py](https://github.com/bradleyboehmke/rossmann-forecasting/blob/main/src/data/validate_data.py) (377 lines) uses 0.x APIs (`RuntimeBatchRequest`, `SimpleCheckpoint`) that 1.x removed, but the course's YouTube pipeline is a working 1.x pattern to copy. Estimated medium effort |

## 4. Reuse opportunities

| Asset | Reuse in the redesign |
|---|---|
| Book chapters 01–03, 05, 10, 12 | Keep as Learn It with light edits and callouts added |
| Book chapters 04, 06–09 | Keep the concept sections as Learn It; rewrite the hands-on walkthroughs against the reference system |
| Chapter 11 | Split: GitHub web interface steps become Week 1 Try It; command-line Git moves to Build It |
| Chapter 12 | Distribute into Human Side callouts and Week 1 and Week 6 content |
| Rossmann maturity branches | Week 1 See It and Apply It (technical debt comparison) |
| Rossmann `main` | Basis for the course template repository |
| Rossmann tests and CI | Week 5 Try It (inspect and diagnose a failing workflow) |
| Rossmann Streamlit Monitoring page and Evidently code | Week 6 See It and Try It |
| Apple-demand ModelOps notebooks | Build It labs and a possible Week 3 supplementary example (small, synthetic, fast) |
| `apple_data.py` drift generator | Model for building controlled drift into the reference system |
| YouTube pipeline | Optional recorded See It only, or replaced |
| Case study links (ch. 01, 05, 12) and `papers/` | Apply It readings |
| [02-before-we-build-exercise.qmd](../slides/02-before-we-build-exercise.qmd) | Week 1 Apply It and team milestone template |
| [grading/2026-fall-final-project-proposal/](../grading/) rubric categories (local only) | Starting point for the Week 2 milestone rubric |

## 5. Gaps relative to the redesign

- No Try It activities, lab checks, or Colab notebooks exist.
- No See It videos are embedded in the book.
- No devcontainer or Codespaces configuration in either repository.
- No Week 0 materials: readiness self-assessment, survey, environment check, AI-use policy.
- No slides for Weeks 5–7.
- No Human Side or Build with AI callouts as named conventions.
- No intentional failure scenarios packaged for students in Rossmann (drift batches exist; invalid data, failing test, and architectural flaw scenarios do not).
- No team project materials for the new structure: survey, charter template, role descriptions, evidence minimum templates, rubrics.
- No Canvas lab check question banks.
- Learning objectives in [README.md](../README.md) are out of date, and its module table (7 modules, monitoring inside Module 4, "Designing MLOps workflows" as Module 6) does not match the new sequence.

## 6. Technical debt and maintenance concerns

| Issue | Evidence | Risk |
|---|---|---|
| Publishing uses Quarto pre-release and a daily cron | [publish-book.yml](../.github/workflows/publish-book.yml) | A pre-release regression can break the book mid-term with no content change |
| Unpinned dependencies in Rossmann | `pyproject.toml` uses `>=` everywhere, no lockfile | Colab or Codespaces installs drift between terms |
| Rossmann declares `evidently>=0.4.0` but uses the 0.7+ API | `drift_detection.py:18-19` | An install resolving to an old version breaks monitoring |
| Course ModelOps uses the legacy Evidently API | [model-drift-monitoring.py:6-8](../ModelOps/model-drift-monitoring.py#L6-L8) (`ColumnMapping`, `metric_preset`) | Cannot be installed next to the Rossmann stack |
| Two Great Expectations versions across assets | DataOps 1.3.1, Rossmann `<1.0` | Students see two incompatible APIs |
| Dockerfile depends on an internal corporate registry | [model-deploy/Dockerfile](../ModelOps/model-deploy/Dockerfile) (`FROM docker-all.artifactory.8451.cloud/...`) | Cannot build outside that network; must change before any student use |
| Committed MLflow store | `ModelOps/mlruns/` (257 files) | Absolute paths and stale run IDs; confusing next to a new tracking setup |
| Large committed binaries | Rossmann `models/final/` 49 MB, `train.csv` 36 MB | Slow clones and Codespace starts |
| Duplicate Great Expectations configuration | Rossmann `great_expectations/` and `gx/` | Unclear which is in use |
| DVC remotes unusable outside the author's machine | Both repositories | `dvc pull` fails for students |
| YouTube API key and transcript scraping | DataOps | Per-student setup; unreliable from cloud |

## 7. Dependencies between components

- Chapters 06–09 depend on `ModelOps/` file paths. Switching to the reference system means updating those walkthroughs and links together with the code.
- The Week 3 ModelOps Try It depends on Week 2 output, Week 4 serving depends on a registered model from Week 3, and Week 6 monitoring depends on Week 4 prediction logs. The template must ship precomputed artifacts at each stage so a student who misses a week can still start the next one.
- The team project's evidence minimums depend on course templates (Colab notebooks per role) that reuse the Try It notebooks.
- Codespaces Try Its depend on a devcontainer in the template repository; local VS Code depends on the same file.
- The publish workflow renders only `book/`, so `planning/`, `docs/adr/`, and `GLOSSARY.md` are not published.
