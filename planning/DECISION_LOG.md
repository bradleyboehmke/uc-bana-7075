# Decision Log

**Approver:** Brad Boehmke (sole approver; Tim Lucas kept informed, not a reviewer).
**Sources:** planning conversations of 2026-10-29 through 2026-11-19, and [revamp_spec.md](../revamp_spec.md).

Status meanings: **Confirmed** = decided by the instructor · **Approved recommendation** = proposed in these planning documents and approved with the plan · **Open** = needs an answer before dependent work starts.

## Confirmed decisions

### Course identity and structure

| ID | Decision | Date |
|---|---|---|
| D-01 | Every student learns what goes into designing *and* developing ML systems; implementation depth varies. The course serves builders and those who will lead builders | 2026-10-30 |
| D-17 | One set of materials for both async and in-class sections, running concurrently. In-class sessions are recorded and shared with the async section | 2026-11-19 |
| D-23 | Week sequence: 0 readiness, 1 architecture, 2 DataOps, 3 ModelOps, 4 serving, 5 DevOps, 6 monitoring, 7 integration and defense. Week 7 includes a short Learn It | 2026-11-19 |
| D-13 | Explainability is not a section, week topic, or required competency. It appears in reference system code and chapter callouts only | 2026-11-19 |
| D-16 | Chapter 12 (human side) and chapter 13 content is redistributed into other modules; a recurring **Human Side callout** carries it. "Human in the loop" keeps its narrow technical meaning | 2026-11-19 |
| D-35 | Vocabulary as defined in [GLOSSARY.md](../GLOSSARY.md), including **design document** for the team's written deliverable | 2026-11-19 |

### Modules and activities

| ID | Decision | Date |
|---|---|---|
| D-03 | Build It is optional and ungraded; no extra credit anywhere in the course | 2026-11-19 |
| D-04 | Each Try It is graded through an individual Canvas **lab check** asking about observed outputs | 2026-11-19 |
| D-19 | Every student completes the Git competency (issue, edit, pull request) in the GitHub web interface, in the team repository | 2026-11-19 |
| D-18 | AI assistants are allowed in all coursework with disclosure. The AI-use policy is a Week 0 pre-read. **Build with AI callouts** appear throughout chapters, with Claude Code demonstration videos | 2026-11-19 |
| D-24 | Week 0 is one completion-graded checklist: AI-use policy, team survey, ungraded readiness self-assessment, environment check | 2026-11-19 |
| D-32 | One pre-produced 5–8 minute See It video per module (eight) before launch, embedded in the book; class recordings cover Try It walkthroughs | 2026-11-19 |

### Example systems

| ID | Decision | Date |
|---|---|---|
| D-06 | One **reference system** followed end to end all term; each module may add one or two **supplementary examples** | 2026-11-19 |
| D-07 | Rossmann is the reference system, confirmed after the options evaluation ([REFERENCE_SYSTEM_OPTIONS.md](REFERENCE_SYSTEM_OPTIONS.md), [ADR 0001](../docs/adr/0001-rossmann-as-reference-system.md)) | 2026-11-19 |
| D-08 | The reference system is a new GitHub template repository derived from Rossmann `main`. The original repository and its maturity branches are untouched. Teams create their repositories from the template | 2026-11-19 |
| D-09 | Single LightGBM model with fixed hyperparameters; ensemble and Optuna tuning move to Build It | 2026-11-19 |
| D-10 | Rossmann's three maturity branches are the Week 1 See It and Apply It | 2026-11-19 |
| D-11 | Keep Great Expectations, upgraded to 1.x | 2026-11-19 |
| D-36 | Simplifications: down-sampled stores, committed sample data instead of the local DVC remote (DVC taught through See It), trimmed features | 2026-11-19 |
| D-12 | Supplementary examples need no API keys or accounts, use public data, and run in Colab in under 5 minutes, or are recorded See It only. At most two per module. The YouTube pipeline is optional and may be replaced | 2026-11-19 |
| D-14 | **Case studies** (read and analyzed) are distinct from supplementary examples (demonstrated) | 2026-11-19 |
| D-15 | The LLM "predict, then explain" feature is deferred until after launch | 2026-11-19 |

### Environments

| ID | Decision | Date |
|---|---|---|
| D-21 | Colab for notebook Try Its; GitHub Codespaces for Try Its that run servers (MLflow UI, API, Streamlit); recorded See It as fallback | 2026-11-19 |
| D-22 | Local VS Code uses the same devcontainer through Dev Containers, with a `uv`/`pip` fallback; supported as self-service only | 2026-11-19 |

### Team project and grading

| ID | Decision | Date |
|---|---|---|
| D-02 | Keep team projects with role-based teams formed by the instructor from a Week 0 survey | 2026-10-30 |
| D-05 | Each team designs its own **team system** on a dataset of its choosing. The reference system is the exemplar, assessed through quizzes, not the project | 2026-11-19 |
| D-20 | Team datasets must be public, tabular, and supervised (classification, regression, forecasting), approved at the Week 2 milestone | 2026-11-19 |
| D-25 | Every member produces **component evidence** for their role on the team's own data. Components may be shown separately; end-to-end integration is not required. Team-level tiers are retired | 2026-11-19 |
| D-26 | Evidence minimums per role as in [COURSE_REDESIGN.md §10](COURSE_REDESIGN.md#10-team-project). Five-person teams add a **reliability lead**; in three-person teams the product lead also produces the drift report. The platform lead may serve a template baseline model if the ML lead's model is late | 2026-11-19 |
| D-27 | Component evidence is graded on correctness, verification, and explanation, not depth; an excellent evidence minimum can earn full marks. Integration is recognized in feedback, not points | 2026-11-19 |
| D-28 | Final deliverable: team design document plus a recorded team presentation with demo, in which each member presents and demos their own component. Each segment and an individual reflection are graded individually | 2026-11-19 |
| D-29 | Weights: design document and milestones 30% (team); component evidence, segment, and reflection 20%; lab checks 25%; Apply It 20%; Week 0 checklist 5% (all individual) | 2026-11-19 |
| D-30 | Canvas peer evaluations in Weeks 4 and 7 inform a discretionary **contribution adjustment** to the team grade; not a weight of their own | 2026-11-19 |

### Planning and launch

| ID | Decision | Date |
|---|---|---|
| D-31 | First offering launches late February or early March 2027. The bulk of the redesign is built for it, then refined over summer 2027 for Fall 2027. Weeks 0–2 complete at launch; Weeks 3–7 need Learn It, Try It, lab checks, and Apply It | 2026-11-19 |
| D-33 | Brad Boehmke is the sole approver | 2026-11-19 |
| D-34 | Planning documents live in `planning/`, ADRs in `docs/adr/`, and the glossary at the repository root; none are published with the book | 2026-11-19 |
| D-37 | The Kaggle Rossmann rules permit reuse of the dataset, so the template ships down-sampled real Rossmann data rather than synthetic data (resolves Q-01, gate G2) | 2026-11-19 |
| D-38 | First day of class is 2027-03-01 (resolves Q-02) | 2026-11-19 |
| D-39 | The reference system template repository lives under the instructor's personal GitHub account (resolves Q-03) | 2026-11-19 |
| D-40 | Help is available for development or testing, including a tester for the pilot (resolves Q-05) | 2026-11-19 |

## Approved recommendations

The instructor approved the full plan, including all of these recommendations, on 2026-11-19 (gate G1).

| ID | Recommendation | Where |
|---|---|---|
| P-01 | Week 2 DataOps is the pilot module, built end to end before any other week | [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) Phase 2 |
| P-02 | Pin a stable Quarto release and drop the daily publish cron | REPO-03 |
| P-03 | Move apple-demand ModelOps code and the YouTube pipeline into a `labs/` folder for Build It; delete the committed `ModelOps/mlruns/` store | REPO-04 |
| P-04 | Reference system ships precomputed artifacts for every stage so students can start any week fresh | REF-06 |
| P-05 | Five packaged failure scenarios: invalid data, failing test, drift, performance degradation, architectural flaw | REF-07 |
| P-06 | "ML for non-modelers" primer as a book appendix | CID-04 |
| P-07 | Team assignment script with balance rules (a builder per team, at most two non-technical students, overlapping time zones) | PROJ-02 |
| P-08 | New Week 7 chapter "Designing ML system workflows"; chapter 13 becomes an appendix | CONT-08 |
| P-09 | Chapter 11 split: web-interface GitHub steps become a Week 1 Try It; command-line Git moves to Week 5 Build It | CONT-06 |
| P-10 | Reference system template repository working name `bana7075-reference-system` | REF-02 |
| P-11 | Candidate keyless DataOps supplementary example: Open-Meteo weather data or a city open-data portal | CASE-03 |
| P-12 | Repository layout in [COURSE_REDESIGN.md §14](COURSE_REDESIGN.md#14-target-course-architecture) | — |

## Open questions

| ID | Question | Blocks | Owner |
|---|---|---|---|
| Q-04 | Which Canvas tool for peer evaluations (built-in peer review, a survey, or a third-party tool)? | ASSESS-04 | Instructor |
| Q-06 | Instructor guide location: in `planning/` (public repository) or private? | VID-03 | Instructor |
| Q-07 | Who are the helpers (D-40), and which tasks do they take? At minimum a non-coder tester for QA-02 in mid-January and the fresh-account dry run (QA-03) in late February | QA-02, QA-03 | Instructor |

Resolved: Q-01 → D-37, Q-02 → D-38, Q-03 → D-39, Q-05 → D-40.
