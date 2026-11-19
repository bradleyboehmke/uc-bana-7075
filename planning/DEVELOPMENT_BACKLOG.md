# Development Backlog

Epics, tasks, dependencies, and acceptance criteria for the redesign. Phases refer to [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Decisions referenced as D-nn are in [DECISION_LOG.md](DECISION_LOG.md).

**Priority:** **P0** blocks launch and blocks other tasks · **P1** required for launch · **P2** deferred to summer 2027 or later.
**Effort:** **S** under a day · **M** 1–3 days · **L** more than 3 days.
**"RS repo"** is the new course template repository derived from Rossmann (REF-02).

## Summary

| Epic | Tasks | P0 | P1 | P2 |
|---|---|---|---|---|
| 1. Course identity and curriculum | CID-01 – CID-04 | 2 | 2 | 0 |
| 2. Repository and content restructuring | REPO-01 – REPO-05 | 2 | 3 | 0 |
| 3. Reference system | REF-01 – REF-10 | 7 | 2 | 1 |
| 4. Student environments | ENV-01 – ENV-04 | 2 | 2 | 0 |
| 5. Weekly instructional content | CONT-01 – CONT-09 | 1 | 8 | 0 |
| 6. Guided practical activities | TRY-01 – TRY-08 | 1 | 7 | 0 |
| 7. Optional engineering labs | BUILD-01 – BUILD-04 | 0 | 2 | 2 |
| 8. AI-assisted development | AI-01 – AI-04 | 0 | 3 | 1 |
| 9. Case studies and teardowns | CASE-01 – CASE-03 | 0 | 3 | 0 |
| 10. Team project | PROJ-01 – PROJ-07 | 1 | 6 | 0 |
| 11. Assessment and grading | ASSESS-01 – ASSESS-04 | 0 | 4 | 0 |
| 12. Documentation and videos | VID-01 – VID-04 | 0 | 3 | 1 |
| 13. Testing and quality assurance | QA-01 – QA-05 | 1 | 3 | 1 |

---

## Epic 1: Course identity and curriculum design

### CID-01 · Approve the planning documents
- **Description:** Review the seven planning documents and ADR 0001; resolve open questions in the decision log.
- **Rationale:** Every later task depends on these decisions.
- **Priority / Effort / Phase:** P0 · S · Phase 0 · **Done 2026-11-19**
- **Dependencies:** none
- **Deliverables:** Approved `planning/` documents; updated [DECISION_LOG.md](DECISION_LOG.md)
- **Acceptance criteria:** Met. Remaining open questions (Q-04, Q-06, Q-07) block only ASSESS-04, VID-03, and QA testing, not Phase 1.
- **Existing files:** `planning/*`, [revamp_spec.md](../revamp_spec.md)

### CID-02 · Rewrite course description and learning objectives
- **Description:** Replace the README's course description, objectives, and module table with those in [COURSE_REDESIGN.md](COURSE_REDESIGN.md) §2, §4, §6.
- **Rationale:** The README and book index are the public face of the course and currently describe the old structure.
- **Priority / Effort / Phase:** P0 · S · Phase 1
- **Dependencies:** CID-01
- **Deliverables:** Updated README and book index
- **Acceptance criteria:** Six objectives and the Week 0–7 table appear in both files; no mention of explainability as a week topic.
- **Existing files:** [README.md](../README.md), [book/index.qmd](../book/index.qmd)

### CID-03 · Syllabus
- **Description:** Write the Spring 2027 syllabus: objectives, schedule, grading weights, AI-use policy reference, team policy, late work.
- **Rationale:** Needed for Canvas and the registrar before launch.
- **Priority / Effort / Phase:** P1 · M · Phase 4
- **Dependencies:** CID-02, ASSESS-01, AI-01
- **Deliverables:** Syllabus document (kept out of git, matching the existing `*Syllabus*` ignore rule)
- **Acceptance criteria:** Weights total 100% and match COURSE_REDESIGN §11; dates match the term calendar.

### CID-04 · "ML for non-modelers" primer
- **Description:** A one-to-two page reference covering training, features, overfitting, validation, and common metrics, linked from Week 0.
- **Rationale:** Requested directly in Fall 2026 feedback.
- **Priority / Effort / Phase:** P1 · S · Phase 2
- **Dependencies:** CID-01
- **Deliverables:** New book appendix page
- **Acceptance criteria:** A reader with no ML background can define each listed term; linked from the readiness self-assessment.
- **New files:** `book/appendix-ml-primer.qmd`

## Epic 2: Repository and content restructuring

### REPO-01 · Reorder and repartition book chapters
- **Description:** Update `_quarto.yml` parts to Weeks 0–7; move monitoring after DevOps; make ch. 13 an appendix; add placeholders for the Week 0 and Week 7 chapters.
- **Rationale:** The book's structure must match the new sequence before content work starts.
- **Priority / Effort / Phase:** P0 · S · Phase 1
- **Dependencies:** CID-01
- **Deliverables:** Updated `_quarto.yml`; book renders
- **Acceptance criteria:** Book renders locally with parts named by week; all cross-references (`@sec-...`) resolve.
- **Existing files:** [book/_quarto.yml](../book/_quarto.yml), [book/13-continued-learning.qmd](../book/13-continued-learning.qmd)
- **New files:** `book/00-orientation.qmd`, `book/14-designing-ml-workflows.qmd`

### REPO-02 · Module page template and callout styles
- **Description:** Define the five-component module structure as a Quarto include or page pattern, and custom callout classes for Human Side and Build with AI.
- **Rationale:** Consistency across modules; writing to a template is faster.
- **Priority / Effort / Phase:** P0 · M · Phase 2
- **Dependencies:** REPO-01
- **Deliverables:** Template, callout CSS, one worked example in the pilot chapter
- **Acceptance criteria:** Both callouts render distinctly in light and dark themes; pilot module (Week 2) uses the template.
- **Existing files:** [book/_quarto.yml](../book/_quarto.yml)
- **New files:** `book/_templates/module.qmd`, `book/styles/callouts.scss`

### REPO-03 · Pin Quarto version and remove daily cron
- **Description:** Pin a stable Quarto release in the publish workflow and drop the daily schedule.
- **Rationale:** Pre-release Quarto can break the book mid-term with no content change.
- **Priority / Effort / Phase:** P1 · S · Phase 1
- **Dependencies:** none
- **Deliverables:** Updated workflow
- **Acceptance criteria:** Workflow uses a fixed stable version; book publishes on push to `main`.
- **Existing files:** [.github/workflows/publish-book.yml](../.github/workflows/publish-book.yml)

### REPO-04 · Retire or relocate legacy code folders
- **Description:** Move the apple-demand ModelOps code and YouTube pipeline into a `labs/` area used only by Build It; remove the committed `mlruns/` store; replace the corporate Dockerfile base image.
- **Rationale:** Avoids students confusing legacy code with the RS; removes unbuildable and stale artifacts.
- **Priority / Effort / Phase:** P1 · M · Phase 3
- **Dependencies:** REF-02, BUILD-01
- **Deliverables:** Reorganized folders; updated links in chapters 04 and 06–09
- **Acceptance criteria:** No chapter links to a moved path; `model-deploy/Dockerfile` builds from a public base image.
- **Existing files:** [DataOps/](../DataOps/), [ModelOps/](../ModelOps/), [model-deploy/Dockerfile](../ModelOps/model-deploy/Dockerfile)

### REPO-05 · Slides for Weeks 0–7
- **Description:** Renumber existing decks to the new weeks and create decks for Weeks 5–7 (currently missing).
- **Rationale:** In-class delivery needs them, and recordings feed the async section.
- **Priority / Effort / Phase:** P1 · L · Phase 4
- **Dependencies:** CONT tasks for the matching week
- **Deliverables:** One deck per week
- **Acceptance criteria:** Deck titles match the Week 0–7 table; each deck renders.
- **Existing files:** [slides/](../slides/)

## Epic 3: Reference ML system

### REF-01 · Verify Kaggle redistribution terms
- **Description:** Read the Rules tab of the Rossmann Store Sales competition while signed in to Kaggle and record whether redistributing the data in a public repository is allowed.
- **Rationale:** Determines whether the template ships real or synthetic data.
- **Priority / Effort / Phase:** P0 · S · Phase 0 · **Instructor task** · **Done 2026-11-19**
- **Dependencies:** none
- **Deliverables:** Answer recorded in the decision log (D-37: reuse permitted)
- **Acceptance criteria:** Met. REF-03 follows path (a).

### REF-02 · Create the course template repository
- **Description:** New repository derived from Rossmann `main`, marked as a GitHub template.
- **Rationale:** Keeps the original Rossmann repository and maturity branches intact for Week 1 (D-08).
- **Priority / Effort / Phase:** P0 · S · Phase 1
- **Dependencies:** CID-01
- **Deliverables:** Template repository
- **Acceptance criteria:** "Use this template" creates a working copy; original Rossmann repository unchanged.
- **New files:** New repository under the instructor's personal GitHub account (D-39; working name `bana7075-reference-system`)

### REF-03 · Down-sample data
- **Description:** Select a store subset of the real Rossmann data and commit it (D-37 confirms reuse is permitted; the synthetic-data path is no longer needed).
- **Rationale:** D-36; the full pipeline must run in a few minutes in Colab.
- **Priority / Effort / Phase:** P0 · M · Phase 1
- **Dependencies:** REF-01, REF-02
- **Deliverables:** Committed data under the size target; generation script if synthetic
- **Acceptance criteria:** Full pipeline runs in under 5 minutes on a free Colab runtime; committed data under 10 MB.
- **Existing files:** Rossmann `data/raw/`

### REF-04 · Simplify the model and features
- **Description:** Replace the ensemble and Optuna tuning with a single LightGBM model with fixed hyperparameters; trim features to a set that fits one diagram; add SHAP in code.
- **Rationale:** D-09, D-13; the system around the model is the subject.
- **Priority / Effort / Phase:** P0 · M · Phase 1
- **Dependencies:** REF-02
- **Deliverables:** Simplified `src/models/`, `config/params.yaml`; updated tests
- **Acceptance criteria:** Training completes in under 2 minutes in Colab; tests pass; one feature diagram covers every feature.
- **Existing files:** Rossmann `src/models/train_ensemble.py`, `train_final.py`, `ensemble.py`, `src/features/build_features.py`, `config/params.yaml`, related tests

### REF-05 · Upgrade Great Expectations to 1.x
- **Description:** Port `validate_data.py` and suites from 0.x APIs to 1.x; consolidate the two configuration directories; keep Data Docs.
- **Rationale:** D-11; aligns with the course's existing 1.3.1 usage.
- **Priority / Effort / Phase:** P0 · M · Phase 1
- **Dependencies:** REF-02
- **Deliverables:** Working validation on 1.x; one config directory
- **Acceptance criteria:** Raw and feature suites run and produce Data Docs; tests pass.
- **Existing files:** Rossmann `src/data/validate_data.py`, `great_expectations/`, `gx/`; reference pattern in [DataOps/dataops_utils.py](../DataOps/dataops_utils.py)

### REF-06 · Replace DVC remote with committed sample data and precomputed stage artifacts
- **Description:** Remove the local DVC remote dependency; commit small artifacts for each stage (processed data, features, a registered model, prediction logs) so any week can start fresh.
- **Rationale:** `dvc pull` fails for students; missing a week must not block the next.
- **Priority / Effort / Phase:** P0 · M · Phase 1
- **Dependencies:** REF-03, REF-04
- **Deliverables:** `artifacts/` with a README describing each stage's snapshot
- **Acceptance criteria:** Each week's Try It starts from committed artifacts alone.
- **Existing files:** Rossmann `dvc.yaml`, `.dvc/config`, `data/processed/*.dvc`

### REF-07 · Package failure scenarios
- **Description:** Add five documented scenarios: invalid-data batch, breaking change that fails a test (on a branch), drifted batch, performance-degradation batch, and a documented architectural flaw.
- **Rationale:** Spec §6; Try Its in Weeks 2, 5, 6 and the Week 7 teardown depend on them.
- **Priority / Effort / Phase:** P0 · M · Phase 2
- **Dependencies:** REF-04, REF-05, REF-06
- **Deliverables:** `scenarios/` folder with data, branch, and a README per scenario
- **Acceptance criteria:** Each scenario produces the documented failure on a clean copy of the template.
- **Existing files:** Rossmann `data/batch_predictions/`, `scripts/generate_batch_prediction_data.py`

### REF-08 · Pin dependencies and add a lockfile
- **Description:** Pin every dependency, generate a `uv` lockfile, and fix the declared Evidently version to match the code.
- **Rationale:** Rossmann uses `>=` throughout and declares `evidently>=0.4.0` while using the 0.7+ API.
- **Priority / Effort / Phase:** P1 · S · Phase 1
- **Dependencies:** REF-04, REF-05
- **Deliverables:** Pinned `pyproject.toml`, `uv.lock`
- **Acceptance criteria:** A fresh install in Colab and Codespaces resolves identical versions.
- **Existing files:** Rossmann `pyproject.toml`

### REF-09 · Architecture diagram and student-facing README
- **Description:** One diagram of the full system and a README written for non-coders, linking each component to its week.
- **Rationale:** Week 1 trace Try It and every See It use it.
- **Priority / Effort / Phase:** P1 · S · Phase 2
- **Dependencies:** REF-04 through REF-07
- **Deliverables:** Diagram, README
- **Acceptance criteria:** Every pipeline stage in the code appears in the diagram.
- **Existing files:** Rossmann `README.md`, `docs/`

### REF-10 · LLM "predict, then explain" feature
- **Description:** Add an LLM-generated plain-language explanation grounded in predictions and SHAP output.
- **Rationale:** Spec §10; deferred by D-15.
- **Priority / Effort / Phase:** P2 · L · after launch
- **Dependencies:** REF-04
- **Acceptance criteria:** Defined when the feature is scheduled.

## Epic 4: Student development environments

### ENV-01 · Devcontainer for Codespaces and local VS Code
- **Description:** A devcontainer in the RS repo that installs pinned dependencies and forwards MLflow, API, and Streamlit ports, with one-command launchers.
- **Rationale:** D-21, D-22.
- **Priority / Effort / Phase:** P0 · M · Phase 1
- **Dependencies:** REF-08
- **Deliverables:** `.devcontainer/devcontainer.json`, launcher tasks
- **Acceptance criteria:** A new Codespace starts and serves all three UIs within 5 minutes; the same file opens locally with Dev Containers.
- **New files:** `.devcontainer/devcontainer.json` in the RS repo
- **Existing files:** Rossmann `scripts/launch_*.sh`, `start_mlflow.sh`

### ENV-02 · Colab notebooks for notebook-based Try Its
- **Description:** One Colab notebook per Colab Try It (Weeks 0, 2, 3, 6) and per role evidence template, with a pinned install cell and an "Open in Colab" badge.
- **Rationale:** D-21; no local setup for required work.
- **Priority / Effort / Phase:** P0 · M · Phase 2–3
- **Dependencies:** REF-06, REF-08
- **Deliverables:** `notebooks/try-it/*.ipynb`, `notebooks/evidence/*.ipynb` in the RS repo
- **Acceptance criteria:** Each notebook runs top to bottom on a fresh free Colab runtime in under 10 minutes.

### ENV-03 · Local setup guide (self-service)
- **Description:** Dev Containers instructions plus a `uv`/`pip` fallback for students without Docker, labeled self-service.
- **Rationale:** D-22.
- **Priority / Effort / Phase:** P1 · S · Phase 3
- **Dependencies:** ENV-01
- **Deliverables:** Guide in the book's Week 0 chapter
- **Acceptance criteria:** Followed successfully on macOS and Windows by a tester.

### ENV-04 · Environment check and fallback artifacts
- **Description:** The Week 0 environment check, plus committed screenshots and HTML reports for every Try It in case a service is unavailable.
- **Rationale:** Spec §9 fallback principle.
- **Priority / Effort / Phase:** P1 · S · Phase 3
- **Dependencies:** ENV-01, ENV-02
- **Deliverables:** Week 0 checklist activity; fallback artifacts linked from each Try It
- **Acceptance criteria:** Every Try It's lab check can be answered from the fallback artifacts if needed.

## Epic 5: Weekly instructional content

One task per week, plus the cross-cutting chapter 12 redistribution. Each covers Learn It, See It script, Apply It, and callouts per [CURRICULUM_MAP.md](CURRICULUM_MAP.md); Try Its are in Epic 6.

| ID | Week | Priority / Effort / Phase | Dependencies | Existing files | New files | Acceptance criteria |
|---|---|---|---|---|---|---|
| CONT-01 | 0 Orientation | P1 · M · 3 | REPO-01, CID-04, AI-01 | [index.qmd](../book/index.qmd), [00-intro.qmd](../slides/00-intro.qmd) | `book/00-orientation.qmd` | Checklist, policy, primer, and environment check linked |
| CONT-02 | 2 DataOps (**pilot**) | P0 · L · 2 | REPO-02, REF-07 | [03-dataops-role.qmd](../book/03-dataops-role.qmd), [04-dataops-build.qmd](../book/04-dataops-build.qmd) | — | All five components present; hands-on section rewritten against the RS; pilot review passed (QA-02) |
| CONT-03 | 1 Architecture | P1 · M · 3 | CONT-02 | [01](../book/01-intro-ml-system.qmd), [02](../book/02-before-we-build.qmd) | — | Maturity-branch content and Human Side callouts added |
| CONT-04 | 3 ModelOps | P1 · L · 3 | CONT-02 | [05](../book/05-modelops-role.qmd), [06](../book/06-modelops-experimenting.qmd), [07](../book/07-modelops-versioning.qmd) | — | 18 `ModelOps/` references rewritten or moved to Build It; explainability callout added |
| CONT-05 | 4 Serving | P1 · L · 4 | CONT-02 | [08](../book/08-modelops-deployment.qmd), [99](../book/99-ml-serving-methods.qmd) | — | 19 `ModelOps/` references handled |
| CONT-06 | 5 DevOps | P1 · M · 4 | CONT-02 | [10](../book/10-devops-role.qmd), [11](../book/11-devops-git.qmd) | — | Ch. 11 split into web-interface Try It and command-line Build It |
| CONT-07 | 6 Monitoring | P1 · L · 4 | CONT-02 | [09](../book/09-modelops-monitoring.qmd) | — | 10 `ModelOps/` references handled; Evidently examples on current API |
| CONT-08 | 7 Integration | P1 · M · 4 | CONT-03 – CONT-07 | [README.md](../README.md) Module 6 topics, [13](../book/13-continued-learning.qmd) | `book/14-designing-ml-workflows.qmd` | Short Learn It and architecture teardown written |
| CONT-09 | Human Side redistribution | P1 · M · 3 | REPO-02 | [12-human-side-of-ml.qmd](../book/12-human-side-of-ml.qmd) | — | Callouts placed in Weeks 1, 4, 6; remaining content kept as a reading; nothing lost |

**Rationale (all):** Learn It content is largely reusable; the work is restructuring and rewriting hands-on sections against the RS. Week 2 goes first as the pilot because it exercises every component and the most failure-scenario tooling.

## Epic 6: Guided practical activities (Try It and lab checks)

| ID | Try It | Env | Priority / Effort / Phase | Dependencies | Acceptance criteria |
|---|---|---|---|---|---|
| TRY-01 | Week 0 environment check | Colab, Codespaces, GitHub | P1 · S · 3 | ENV-01, ENV-02 | Completes in under 20 minutes for a non-coder tester |
| TRY-02 | Week 2 invalid-data batch (**pilot**) | Colab | P0 · M · 2 | REF-07, ENV-02 | Non-coder tester completes it and answers the lab check correctly |
| TRY-03 | Week 1 trace a prediction; GitHub web PR | Codespaces, GitHub | P1 · M · 3 | REF-09, ENV-01 | PR created from the browser only |
| TRY-04 | Week 3 compare runs and promote champion | Colab + Codespaces | P1 · M · 3 | REF-06, ENV-01 | Runs appear in the MLflow UI; alias changes |
| TRY-05 | Week 4 API and batch requests | Codespaces | P1 · M · 4 | ENV-01 | Valid and invalid requests produce the documented responses |
| TRY-06 | Week 5 failing CI run | GitHub, Codespaces | P1 · M · 4 | REF-07 | Packaged change fails the documented test in Actions |
| TRY-07 | Week 6 drifted and degraded batches | Colab | P1 · M · 4 | REF-07, ENV-02 | Reports show the documented drift and degradation |
| TRY-08 | Lab check question banks for all Try Its | Canvas | P1 · M · 3–4 | TRY-01 – TRY-07 | Each bank has observation questions answerable only by running the activity, plus one explanation question; answer keys written |

**Deliverables (all):** Book section per Try It, notebook or Codespaces steps, lab check bank, fallback artifact.
**Rationale (all):** Required hands-on minimum (spec §2, §8); lab checks make it individually gradable (D-04).

## Epic 7: Optional engineering labs (Build It)

### BUILD-01 · Reframe existing notebooks as Build It labs
- **Description:** Move apple-demand ModelOps notebooks and the YouTube pipeline into Build It sections with short intros; upgrade the Evidently code to the current API.
- **Rationale:** D-03; reuse existing content.
- **Priority / Effort / Phase:** P1 · M · Phase 3–4
- **Dependencies:** REPO-04
- **Deliverables:** Build It sections linking the labs
- **Acceptance criteria:** Each lab runs with the pinned environment; links resolve.
- **Existing files:** [ModelOps/](../ModelOps/), [DataOps/](../DataOps/)

### BUILD-02 · Build It prompts against the RS
- **Description:** For each week, one short extension task on the RS (custom expectation, new endpoint, new test, retraining trigger).
- **Rationale:** Gives builders depth on the same system and supports deeper evidence.
- **Priority / Effort / Phase:** P1 · M · Phase 3–4
- **Dependencies:** REF-07
- **Acceptance criteria:** Each prompt has a stated goal and a way to verify it worked.

### BUILD-03 · Full Build It labs with solutions
- **Description:** Guided multi-step labs with reference solutions on a solutions branch.
- **Priority / Effort / Phase:** P2 · L · summer 2027
- **Dependencies:** BUILD-02

### BUILD-04 · Ensemble and tuning lab
- **Description:** Restore the Rossmann ensemble and Optuna tuning as an advanced lab.
- **Rationale:** D-09 moved them out of the RS.
- **Priority / Effort / Phase:** P2 · M · summer 2027
- **Existing files:** Rossmann `src/models/train_ensemble.py`, `ensemble.py`, `notebooks/04b-hyperparameter-tuning.ipynb`

## Epic 8: AI-assisted development integration

### AI-01 · AI-use policy
- **Description:** Course AI-use policy as a Week 0 pre-read: allowed uses, disclosure format, verification responsibility.
- **Rationale:** D-18.
- **Priority / Effort / Phase:** P1 · S · Phase 2
- **Dependencies:** CID-01
- **Deliverables:** Policy page in the book; disclosure template
- **Acceptance criteria:** Disclosure format used in the design document and reflection templates.
- **Existing files:** Policy from other courses (instructor)

### AI-02 · Build with AI callouts
- **Description:** One or more callouts per module per [CURRICULUM_MAP.md](CURRICULUM_MAP.md), each with use case, example, and risk.
- **Priority / Effort / Phase:** P1 · M · Phase 2–4 (with each CONT task)
- **Dependencies:** REPO-02
- **Acceptance criteria:** Every module has at least one; each names a verification step.

### AI-03 · Claude Code demonstration videos
- **Description:** Short recordings showing an assistant explaining code, debugging a failure scenario, and writing a test.
- **Priority / Effort / Phase:** P1 · M · Phase 3–4
- **Dependencies:** REF-07, VID-01
- **Acceptance criteria:** At least three videos, each under 8 minutes, linked from callouts.

### AI-04 · AI-assisted engineering rubric criteria
- **Description:** Rubric language rewarding evidence of verification of AI-generated work.
- **Priority / Effort / Phase:** P2 · S · summer 2027 (initial language folded into ASSESS-02 for launch)

## Epic 9: Case studies and system teardowns

### CASE-01 · Select and package case studies
- **Description:** Choose one case study per Apply It week from existing links and papers; write discussion prompts.
- **Priority / Effort / Phase:** P1 · M · Phase 3
- **Dependencies:** CONT-02
- **Existing files:** [papers/bernardi2019.pdf](../papers/bernardi2019.pdf), [papers/NIPS-2015-hidden-technical-debt-in-machine-learning-systems-Paper.pdf](../papers/NIPS-2015-hidden-technical-debt-in-machine-learning-systems-Paper.pdf), case links in chapters 01, 05, 12
- **Acceptance criteria:** Each Apply It week has a reading, prompts, and a grading guide.

### CASE-02 · Teardown template
- **Description:** A reusable template: where does this week's concept appear, why is it designed this way, what breaks without it.
- **Priority / Effort / Phase:** P1 · S · Phase 2 (used in the pilot)
- **Acceptance criteria:** Used for Weeks 1, 5, 7 teardowns.

### CASE-03 · Supplementary examples
- **Description:** Select and build the keyless DataOps example (candidate: Open-Meteo or a city open-data portal); confirm the other weeks' supplementary examples in the curriculum map.
- **Rationale:** D-12 criteria; the YouTube pipeline fails them.
- **Priority / Effort / Phase:** P1 · M · Phase 2
- **Dependencies:** none
- **Acceptance criteria:** Each example meets the D-12 criteria (no keys or accounts, public data, under 5 minutes in Colab, or See It only).

## Epic 10: Team project redesign

| ID | Task | Priority / Effort / Phase | Dependencies | Deliverables | Acceptance criteria |
|---|---|---|---|---|---|
| PROJ-01 | Week 0 team formation survey | P0 · S · 2 | CID-01 | Canvas survey | Captures experience, program, time zone, availability, role preferences |
| PROJ-02 | Team assignment rules and script | P1 · S · 3 | PROJ-01 | Script that drafts teams from the survey export | Every team has a builder, at most two non-technical students, overlapping time zones; handles teams of 3 and 5 |
| PROJ-03 | Team charter template | P1 · S · 3 | — | Template | Covers channel, cadence, decision rules, non-participation |
| PROJ-04 | Role descriptions and evidence minimum templates | P1 · M · 3 | ENV-02, TRY-02 – TRY-07 | One page per role; Colab evidence notebook per role | Each template runs on a sample non-Rossmann tabular dataset |
| PROJ-05 | Design document template | P1 · S · 3 | CID-01 | Template with the spec §11 sections and per-role sections | Every role's section is identifiable for individual grading |
| PROJ-06 | Milestone instructions (Weeks 1, 2, 4, 6, 7) | P1 · M · 4 | PROJ-05 | Instructions per milestone | Dataset approval criteria (public, tabular, supervised) stated at Week 2 |
| PROJ-07 | Team repository from the RS template | P1 · S · 3 | REF-02 | Instructions for creating team repos | Used by the Week 1 Git Try It |

**Rationale (all):** D-05, D-19, D-20, D-24 – D-28. **Existing files:** [slides/02-before-we-build-exercise.qmd](../slides/02-before-we-build-exercise.qmd); local `grading/2026-fall-final-project-proposal/` rubric categories.

## Epic 11: Assessment and grading

| ID | Task | Priority / Effort / Phase | Dependencies | Acceptance criteria |
|---|---|---|---|---|
| ASSESS-01 | Grading scheme in Canvas | P1 · S · 4 | CID-01 | Weights 30/20/25/20/5; Week 0 completion-graded |
| ASSESS-02 | Rubrics: design document, milestones, component evidence (correctness, verification, explanation), presentation segment, reflection | P1 · M · 4 | PROJ-04, PROJ-05 | An excellent evidence minimum can earn full marks; no points for depth or integration |
| ASSESS-03 | Apply It grading guides | P1 · S · 3–4 | CASE-01 | One guide per Apply It |
| ASSESS-04 | Peer evaluation and contribution adjustment procedure | P1 · S · 4 | — | Canvas peer evaluations in Weeks 4 and 7; written criteria for applying an adjustment |

## Epic 12: Documentation and instructional videos

### VID-01 · Recording setup and style guide
- **Description:** Recording tool, resolution, captioning, length target (5–8 minutes), and naming.
- **Priority / Effort / Phase:** P1 · S · Phase 2

### VID-02 · Eight See It videos (Weeks 0–7)
- **Description:** One See It video per module, embedded in the book.
- **Priority / Effort / Phase:** P1 · L · Phase 2–4 (about one every 1.5 weeks)
- **Dependencies:** VID-01, the matching REF and CONT tasks
- **Acceptance criteria:** Captioned, 5–8 minutes, embedded and playing in the published book.

### VID-03 · Instructor guide
- **Description:** Week-by-week notes for in-class delivery: which Try It or Build It to walk through live, timing, recording reminders.
- **Priority / Effort / Phase:** P1 · S · Phase 4
- **New files:** `planning/instructor-guide.md` or a private location (instructor choice)

### VID-04 · Additional videos (Try It walkthroughs, Build It)
- **Priority / Effort / Phase:** P2 · L · summer 2027

## Epic 13: Testing and course quality assurance

| ID | Task | Priority / Effort / Phase | Dependencies | Acceptance criteria |
|---|---|---|---|---|
| QA-01 | RS CI: tests plus a scheduled "fresh install" job that runs every Colab notebook | P0 · M · 1–2 | REF-08, ENV-02 | Job fails when an install or notebook breaks |
| QA-02 | Pilot review of Week 2 | P1 · S · 2 | CONT-02, TRY-02 | Review by the instructor and at least one non-coder tester; findings applied to the template before other weeks |
| QA-03 | End-to-end dry run of all Try Its on fresh accounts | P1 · M · 5 | All TRY tasks | Every Try It completed in Colab and Codespaces from a new GitHub account |
| QA-04 | Book link and render check | P1 · S · 5 | All CONT tasks | No broken internal links; book renders and publishes |
| QA-05 | Post-term review | P2 · S · summer 2027 | Launch | Student feedback and lab check results reviewed; backlog for Fall 2027 |
