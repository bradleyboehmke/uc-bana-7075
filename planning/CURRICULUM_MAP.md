# Curriculum Map

Each week's module, what it teaches, which existing content it draws on, and how much work that content needs. Objectives O1–O6 are defined in [COURSE_REDESIGN.md §4](COURSE_REDESIGN.md#4-learning-objectives). "RS" means the reference system (simplified Rossmann template).

**Revision levels:** *Keep* = use as is or with light edits; *Minor* = edits, callouts, and link updates; *Substantial* = rewrite of hands-on sections against the RS; *New* = does not exist yet.

## Coverage at a glance

| Week | O1 Understand | O2 Design | O3 Operate | O4 Evaluate | O5 Troubleshoot | O6 Collaborate | Project milestone |
|---|---|---|---|---|---|---|---|
| 0 | | | ● | | | ● | Team formation survey |
| 1 | ● | ● | ● | ● | | ● | Team charter |
| 2 | ● | ● | ● | ● | ● | ● | Problem, requirements, dataset approval, initial architecture |
| 3 | ● | ● | ● | ● | ● | | |
| 4 | ● | ● | ● | ● | ● | ● | Data, model, and experiment design; peer evaluation 1 |
| 5 | ● | | ● | ● | ● | ● | |
| 6 | ● | ● | ● | ● | ● | ● | Deployment, testing, and monitoring design |
| 7 | ● | ● | | ● | | ● | Design document and recorded presentation; peer evaluation 2 |

## Week 0: Readiness and orientation

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | Course overview, how modules work, AI-use policy pre-read | [index.qmd](../book/index.qmd), [slides/00-intro.qmd](../slides/00-intro.qmd) | Minor (index), New (policy) |
| See It | Video: tour of the RS running end to end | — | New |
| Try It | Environment check: open the RS Colab notebook, start a Codespace, confirm a GitHub account | — | New |
| Apply It | Team formation survey; readiness self-assessment routing to Python and ML primers ("ML for non-modelers" one-pager) | — | New |
| Build It | Local VS Code setup with Dev Containers (self-service) | — | New |
| Graded | Week 0 checklist (completion, 5%) | | |

## Week 1: ML system architecture and foundations

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | ML system design, design principles, lifecycle; stakeholders and whether ML is needed; requirements | [01-intro-ml-system.qmd](../book/01-intro-ml-system.qmd), [02-before-we-build.qmd](../book/02-before-we-build.qmd) | Minor |
| Human Side | Stakeholder alignment and team collaboration (from ch. 12 "Collaboration Between Teams", "The Disconnect Between Technical and Business Teams") | [12-human-side-of-ml.qmd](../book/12-human-side-of-ml.qmd) | Minor (move) |
| See It | Video: the three Rossmann maturity branches; tracing one prediction through the RS architecture | Rossmann branches `1-initial-exploration`, `2-research-mature`, `main` | New |
| Try It | Trace a prediction: open the RS architecture diagram and Streamlit app, follow one store's forecast from raw data to output. Then the Git competency: create an issue, edit a file, and open a pull request in the team repository through the GitHub web interface | [11-devops-git.qmd](../book/11-devops-git.qmd) PR sections | New (trace), Substantial (Git, rewritten for the web interface) |
| Lab check | Which stages did the record pass through? What does the PR change? | | New |
| Apply It | Technical debt comparison across the three maturity branches, using the Hidden Technical Debt paper; Booking.com case study | [papers/](../papers/), [02-before-we-build-exercise.qmd](../slides/02-before-we-build-exercise.qmd) | Minor |
| Build It | Clone and run the RS locally | Rossmann docs `getting-started/` | Minor |
| Build with AI | Use an assistant to explain an unfamiliar repository's structure | — | New |
| Supplementary example | Case studies from ch. 01 (Netflix, Uber Michelangelo, Spotify, Airbnb Bighead, Google TFX) | [01-intro-ml-system.qmd](../book/01-intro-ml-system.qmd) | Keep |

## Week 2: DataOps

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | Ingestion, processing, sampling, missing data, validation, versioning; ETL vs ELT; DataOps tools | [03-dataops-role.qmd](../book/03-dataops-role.qmd), [04-dataops-build.qmd](../book/04-dataops-build.qmd) concept sections | Minor |
| See It | Video: RS pipeline run and Great Expectations Data Docs | Rossmann `src/data/`, `dvc.yaml` | New |
| Try It (Colab) | Run the RS pipeline; load the invalid-data batch; read the failing validation result; explain what would have happened downstream without it | RS failure scenario | New |
| Lab check | Which expectation failed, on which column, and why it matters | | New |
| Apply It | Design data quality checks for a short scenario; data versioning tradeoffs (DVC as See It only) | ch. 03 exercise | Minor |
| Build It | Write a custom expectation; YouTube pipeline walkthrough | [04-dataops-build.qmd](../book/04-dataops-build.qmd) hands-on section, [DataOps/](../DataOps/) | Substantial (hands-on section rewritten against the RS; YouTube kept as optional Build It or replaced) |
| Build with AI | Use an assistant to draft validation rules from a data dictionary, then check them against real data | — | New |
| Supplementary example | Keyless public-data pipeline (candidate: Open-Meteo weather or a city open-data portal), replacing the YouTube pipeline as the second DataOps example | — | New |

## Week 3: ModelOps

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | ModelOps components, experiment tracking, versioning and registry | [05-modelops-role.qmd](../book/05-modelops-role.qmd), [06-modelops-experimenting.qmd](../book/06-modelops-experimenting.qmd), [07-modelops-versioning.qmd](../book/07-modelops-versioning.qmd) concept sections | Minor |
| Callout | Explainability: what SHAP shows and where it fits in a system (callout only) | RS code | New |
| See It | Video: MLflow UI with RS runs, comparing runs, registering a champion | Rossmann `src/models/`, `mlflow_utils.py` | New |
| Try It (Colab + Codespaces) | Run two training configurations in Colab; open the MLflow UI in a Codespace; compare; promote one to `champion` | RS | New |
| Lab check | Which run won on which metric; what else would you check before promoting it | | New |
| Apply It | Uber Michelangelo case study: how a platform standardizes experiment tracking | [05-modelops-role.qmd](../book/05-modelops-role.qmd) case study | Minor |
| Build It | Apple-demand experiment, registration, and retrieval notebooks | [ModelOps/model-experimentation.ipynb](../ModelOps/model-experimentation.ipynb) and related | Minor (reframed as Build It; links updated) |
| Build with AI | Ask an assistant to explain why two runs differ; verify against logged parameters | — | New |
| Supplementary example | Apple-demand MLflow notebooks (small, synthetic, fast) | [ModelOps/](../ModelOps/) | Keep |

Hands-on sections of chapters 06 and 07 (18 references to `ModelOps/`) move to Build It or are rewritten against the RS: *Substantial*.

## Week 4: Model serving and deployment

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | Deployment considerations, strategies, batch vs real-time serving | [08-modelops-deployment.qmd](../book/08-modelops-deployment.qmd), [99-ml-serving-methods.qmd](../book/99-ml-serving-methods.qmd) | Minor (concepts), Substantial (hands-on, 19 `ModelOps/` references) |
| Human Side | Human-in-the-loop review of low-confidence predictions (from ch. 12) | [12-human-side-of-ml.qmd](../book/12-human-side-of-ml.qmd) | Minor (move) |
| See It | Video: RS FastAPI docs page, Streamlit Predictions page, batch scoring | Rossmann `deployment/` | New |
| Try It (Codespaces) | Start the API and app; send valid and invalid requests; run a batch job; compare latency and output | RS | New |
| Lab check | What the API returned for a malformed request and why; batch vs real-time for a given scenario | | New |
| Apply It | Choose a serving pattern for three business scenarios and justify it | ch. 99 criteria | Minor |
| Build It | Add an endpoint to the RS API; apple-demand FastAPI and Docker example | [ModelOps/fastapi_app.py](../ModelOps/fastapi_app.py), [model-deploy/](../ModelOps/model-deploy/) | Minor (Dockerfile base image must be replaced) |
| Build with AI | Generate a request client from the API's OpenAPI schema and test it | — | New |

## Week 5: DevOps and system reliability

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | DevOps for ML, CI/CD, automated testing | [10-devops-role.qmd](../book/10-devops-role.qmd), [11-devops-git.qmd](../book/11-devops-git.qmd) Actions section | Minor |
| See It | Video: RS test suite and a GitHub Actions run passing and failing | Rossmann `tests/`, `.github/workflows/test.yml` | New |
| Try It (GitHub web + Codespaces) | Open a PR in the team repo containing the packaged breaking change; read the failed Actions log; identify the failing test; explain the fix | RS failure scenario | New |
| Lab check | Which test failed, what it protects against, what the fix is | | New |
| Apply It | Hidden Technical Debt analysis (existing ch. 10 exercise) applied to the RS | [10-devops-role.qmd](../book/10-devops-role.qmd) exercise | Keep |
| Build It | Write a new test; add a CI step; command-line Git | [11-devops-git.qmd](../book/11-devops-git.qmd) command-line sections | Minor (moved to Build It) |
| Build with AI | Ask an assistant to write a test, then check that it fails when it should | — | New |
| Supplementary example | Rossmann `2-research-mature` branch: tests without deployment | Rossmann | Keep |

## Week 6: Monitoring and operations

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | Performance, data drift, concept drift, operational monitoring, retraining | [09-modelops-monitoring.qmd](../book/09-modelops-monitoring.qmd) concept sections | Minor (concepts), Substantial (hands-on, 10 `ModelOps/` references) |
| Human Side | Responsible AI, fairness, trust in ML decisions (from ch. 12, including the Amazon hiring example) | [12-human-side-of-ml.qmd](../book/12-human-side-of-ml.qmd) | Minor (move) |
| See It | Video: RS Streamlit Monitoring page and Evidently reports | Rossmann `src/monitoring/`, Monitoring page | New |
| Try It (Colab) | Score the drifted batch and the degraded batch; generate Evidently reports; decide whether to retrain, investigate, or roll back | RS failure scenarios | New |
| Lab check | Which features drifted; whether performance dropped; the recommended action | | New |
| Apply It | Monitoring plan for a case study system | — | New |
| Build It | Implement a retraining trigger; apple-demand drift generator notebook | [ModelOps/model-drift-monitoring.ipynb](../ModelOps/model-drift-monitoring.ipynb), [apple_data.py](../ModelOps/apple_data.py) | Substantial (legacy Evidently API must be upgraded) |
| Build with AI | Ask an assistant to summarize a drift report for a business audience; check every claim against the report | — | New |
| Supplementary example | Apple-demand drift generator with adjustable drift factors | [apple_data.py](../ModelOps/apple_data.py) | Minor |

## Week 7: System integration and defense

| Component | Content | Source | Revision |
|---|---|---|---|
| Learn It | Designing MLOps workflows end to end; infrastructure and scalability; continued learning | [README.md](../README.md) Module 6 topics (no chapter exists), [13-continued-learning.qmd](../book/13-continued-learning.qmd) | New (short chapter), Minor (ch. 13 becomes an appendix) |
| See It | Video: full RS architecture review, with the documented architectural flaw | RS | New |
| Try It | None; time goes to the project | | |
| Apply It | Architecture teardown of the RS's documented flaw | RS failure scenario | New |
| Project | Design document; recorded team presentation with one graded segment per member; individual reflection | — | New |

## Competencies to Try Its

| Competency ([COURSE_REDESIGN.md §8](COURSE_REDESIGN.md#8-required-practical-competencies)) | Week | Try It |
|---|---|---|
| Architecture | 1 | Trace a prediction |
| Git and GitHub | 1 | Issue, edit, pull request in the web interface |
| DataOps | 2 | Invalid-data batch |
| ModelOps | 3 | Compare runs and promote a champion |
| Deployment | 4 | API and batch requests |
| DevOps | 5 | Failing CI run |
| Monitoring | 6 | Drifted and degraded batches |

## Role evidence to Try Its

Each role's evidence minimum repeats a Try It on the team's own data, using the same notebooks with the data source swapped.

| Role | Evidence minimum | Builds on |
|---|---|---|
| Product lead | Baseline predictions against the business metric; risk review | Weeks 1, 6 |
| Data lead | Validation suite with one deliberately failing check | Week 2 |
| ML lead | Two logged experiments, comparison, registered model | Week 3 |
| Platform lead | Served predictions and a drift report | Weeks 4, 6 |
| Reliability lead | Tests and CI on the team repository | Week 5 |

## Chapter disposition

| Chapter | Becomes |
|---|---|
| 01, 02 | Week 1 Learn It |
| 03, 04 | Week 2 Learn It; ch. 04 hands-on becomes Build It |
| 05, 06, 07 | Week 3 Learn It; hands-on sections rewritten or moved to Build It |
| 08, 99 | Week 4 Learn It; hands-on rewritten |
| 10, 11 | Week 5 Learn It; ch. 11 split between Week 1 Try It (web interface) and Week 5 Build It (command line) |
| 09 | Week 6 Learn It; **moves after DevOps** in [_quarto.yml](../book/_quarto.yml) |
| 12 | Split into Human Side callouts in Weeks 1, 4, 6; remainder kept as a reading |
| 13 | Appendix |
| New | Week 0 orientation; Week 7 "Designing ML system workflows" |
