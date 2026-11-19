# BANA 7075 Course Redesign

**Status:** Approved 2026-11-19. Implementation starts with Phase 1 in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md).
**Approver:** Brad Boehmke (sole approver; see [DECISION_LOG.md](DECISION_LOG.md)).
**First offering:** Spring 2027, first day of class 2027-03-01, with in-class and async sections running concurrently. Refinement over summer 2027 before the Fall 2027 offering.

This document states the vision, instructional philosophy, and requirements for the redesign. It supersedes the corresponding sections of [`revamp_spec.md`](../revamp_spec.md) where the decisions made during planning changed them; those changes are listed at the end. Terms in **bold** are defined in [GLOSSARY.md](../GLOSSARY.md).

## 1. Why the course is changing

BANA 7075 now enrolls three kinds of students: M.S. Business Analytics students with Python and ML skills, and a growing number of AI Certificate and MBA students without them. Fall 2026 feedback (section 001, 14 of 49 responding) praised the textbook, the end-to-end project, and the business framing, and asked for more demonstrations, guided practice before the project, less setup friction, and fairer, better-organized group work.

The redesign keeps the course technical. It changes how students reach the technical material: everyone runs, inspects, and breaks a real ML system, and independent software development becomes optional depth rather than the entry requirement.

## 2. Course identity

The course develops the technical literacy and practical competencies to design, evaluate, operate, and improve end-to-end predictive ML systems. It serves students who will build these systems and students who will lead, manage, or buy them; the conviction behind it is that leaders manage ML teams better when they have operated the systems themselves.

The subject is the system around the model, not modeling. Model explainability, LLM features, RAG, and agents are out of scope except as brief mentions; a separate proposed Applied AI Engineering course covers generative AI systems.

## 3. Instructional philosophy

1. **Common conceptual core.** Every student learns the concepts, architectural patterns, engineering practices, and business considerations.
2. **Common hands-on minimum.** Every student runs provided workflows, inspects outputs, modifies configurations, introduces controlled failures, and explains what happened.
3. **Optional deeper engineering.** Students who want it can implement and extend system components.

There are no separate tracks. The progression is **Understand → Run → Observe → Explain → Modify → Build**; everything through Modify is required.

## 4. Learning objectives

All students will be able to:

1. **Understand:** Explain the components, dependencies, and lifecycle of an end-to-end ML system.
2. **Design:** Design ML system architectures that align with business requirements and technical constraints.
3. **Operate:** Execute and interact with data pipelines, modeling workflows, inference services, and monitoring tools.
4. **Evaluate:** Assess system performance, reliability, maintainability, and business value.
5. **Troubleshoot:** Diagnose common system failures and recommend corrective actions.
6. **Collaborate:** Communicate and defend technical decisions within a cross-functional ML team.

Students pursuing optional depth will also implement and extend ML systems using modern software engineering and MLOps practices.

These replace the five objectives in the current [README.md](../README.md), which include "Gain hands-on experience with MLOps tooling" without distinguishing operating from building.

## 5. Module structure

Every **module** has five components, each mapped to book content:

| Component | Required | Purpose | Graded through |
|---|---|---|---|
| **Learn It** | Yes | Concepts, patterns, tradeoffs | Apply It, lab checks |
| **See It** | Yes | A 5–8 minute video and screenshots showing the concept in the **reference system** | Not graded |
| **Try It** | Yes | Guided, low-code activity in a browser environment | **Lab check** (Canvas quiz) |
| **Apply It** | Yes | Design decisions, **case studies**, teardowns, technical debt assessments | Individual Apply It analyses |
| **Build It** | No | Implementation labs reusing existing notebooks and reference system code | Not graded; may feed **deeper evidence** |

Two recurring callouts run through the chapters:

- **Human Side callout:** people, organizational, and ethical considerations. Absorbs most of chapter 12.
- **Build with AI callout:** how an AI coding assistant helps with the topic, a concrete example (often a Claude Code video), and a risk to verify.

AI assistants are allowed in all coursework with disclosure. The AI-use policy is a Week 0 pre-read. Lab checks ask about outputs the student observed, which an assistant cannot supply.

## 6. Curriculum

| Week | Module | Central design question |
|---|---|---|
| 0 | Readiness and orientation | Are you set up, and what do you already know? |
| 1 | ML system architecture and foundations | What are we building, and what business outcome does it support? |
| 2 | DataOps | How do we ensure trustworthy data flows into our system? |
| 3 | ModelOps | How do we develop, compare, reproduce, and register models? |
| 4 | Model serving and deployment | How should predictions be delivered to users and downstream systems? |
| 5 | DevOps and system reliability | How do we safely introduce changes without breaking the system? |
| 6 | Monitoring and operations | How do we know when a deployed system stops performing? |
| 7 | System integration and defense | Can we demonstrate and defend the complete system? |

Week-by-week activities and content mappings are in [CURRICULUM_MAP.md](CURRICULUM_MAP.md).

**Week 0** is one completion-graded checklist: read the AI-use policy, complete the team-formation survey, take an ungraded readiness self-assessment that routes to Python and ML primers, and pass an environment check (open a Colab notebook, start a Codespace, create or confirm a GitHub account).

## 7. Example systems

- **Reference system:** a simplified, course-specific template repository derived from [rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting) `main` (single LightGBM model, Great Expectations 1.x, down-sampled stores, committed sample data, a trimmed feature set, SHAP in code only). The original repository and its three maturity branches stay untouched for Week 1. Selection rationale: [REFERENCE_SYSTEM_OPTIONS.md](REFERENCE_SYSTEM_OPTIONS.md) and [ADR 0001](../docs/adr/0001-rossmann-as-reference-system.md).
- **Supplementary examples:** at most two per module. They need no API keys or accounts, use public data, and run in Colab in under five minutes, or are shown only as a recorded See It.
- **Case studies:** published industry systems (Booking.com, Uber Michelangelo, Netflix, and others) that students read and analyze in Apply It.

The reference system must include intentional failure scenarios: invalid input data, a failing test, deteriorating performance, drifted data, and a problematic architectural decision.

## 8. Required practical competencies

Every student individually demonstrates each of these through a Try It and its lab check.

| Domain | Required competency | Optional depth (Build It) |
|---|---|---|
| Architecture | Trace data and predictions through a working system | Implement a system component |
| Git and GitHub | Create an issue, edit a file, and open a pull request in the GitHub web interface, in the team repository | Branching strategies and automated workflows |
| DataOps | Run a pipeline, introduce invalid data, diagnose the failure | Build pipelines and custom validation checks |
| ModelOps | Run experiments, compare results, justify a model selection | Build training and registration workflows |
| Deployment | Submit inference requests and evaluate responses | Build or extend inference APIs |
| DevOps | Inspect automated tests and diagnose a workflow failure | Write tests and implement CI/CD |
| Monitoring | Interpret drift and performance reports and recommend actions | Implement monitors and retraining triggers |

Explainability is not a required competency. It appears in the reference system code and in chapter callouts only.

## 9. Environments

| Environment | Used for | Support level |
|---|---|---|
| Google Colab | Notebook Try Its: pipelines, validation, training runs, drift reports | Supported |
| GitHub Codespaces | Try Its that run servers: MLflow UI, FastAPI service, Streamlit app | Supported |
| Local VS Code (Dev Containers, same devcontainer as Codespaces; `uv`/`pip` fallback) | Students who prefer local work | Self-service |
| Recorded See It videos and committed artifacts | Fallback when a hosted service is unavailable | Always available |

GitHub Free personal accounts include 120 compute hours and 15 GB-month of Codespaces storage per month ([GitHub docs](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces), checked 2026-11-19). On a 2-core machine that is about 60 hours a month, well above what the Try Its need, provided students stop idle codespaces.

Principles: no required local setup; GitHub introduced progressively; command line optional; dependencies pinned; no hosting services beyond Colab, GitHub, and Codespaces.

## 10. Team project

Teams are formed by the instructor from the Week 0 survey and have defined roles: **product lead**, **data lead**, **ML lead**, **platform lead**, plus a **reliability lead** in five-person teams. In three-person teams the product lead also produces the drift report.

Each team designs a **team system** for a business problem of its choice on a public, tabular, supervised prediction dataset (classification, regression, or forecasting), approved at the Week 2 milestone. The reference system is the exemplar of good architecture and implementation; students are quizzed on it but do not submit it as their project.

Every member produces **component evidence** for their role on the team's own data, meeting at least that role's **evidence minimum**:

| Role | Evidence minimum |
|---|---|
| Product lead | Baseline predictions evaluated against the team's business success metric; risk and responsible AI review of real outputs |
| Data lead | Validation suite run on the team's data, including one deliberately failing check with an explanation |
| ML lead | At least two experiments logged to MLflow, compared, a justified selection, and a registered model |
| Platform lead | Predictions served from the registered model (batch or endpoint) and one drift report comparing two slices of the team's data |
| Reliability lead | Tests and a CI workflow running on the team repository |

If the ML lead's model is late, the platform lead may serve a baseline model trained from the course template. Components may be shown separately; end-to-end integration is recognized in feedback but earns no points. Component evidence is graded on correctness, verification, and explanation, not depth, so an excellent evidence minimum can earn full marks.

**Milestones:** Week 1 team charter; Week 2 business problem, requirements, dataset approval, initial architecture; Week 4 data, model, and experiment design; Week 6 deployment, testing, and monitoring design; Week 7 final **design document** and a recorded team presentation with demo in which each member presents and demos their own component.

## 11. Assessment

| Component | Weight | Individual or team |
|---|---|---|
| Design document and milestones | 30% | Team |
| Component evidence, presentation segment, and reflection | 20% | Individual |
| Lab checks (Try It) | 25% | Individual |
| Apply It analyses (teardowns, case studies) | 20% | Individual |
| Week 0 checklist | 5% | Individual, completion-graded |

Two Canvas peer evaluations (Weeks 4 and 7) inform a **contribution adjustment** applied at the instructor's discretion to a consistently flagged student's share of the team grade. Peer ratings are not a weight of their own. Build It work is ungraded and there is no extra credit.

## 12. Delivery

One set of materials serves both sections. In-class sessions are recorded and shared with the async section; in-class time can walk through Try It, Build It, or team discussions. In addition, each module has at least one pre-produced 5–8 minute See It video embedded in the book (eight before launch).

## 13. Launch scope

Required for launch:

- Weeks 0–2 complete in every component.
- Weeks 3–7 with Learn It, Try It, lab checks, and Apply It complete; See It videos for all eight weeks finished before launch.
- Build It reuses existing notebooks and reference system code with light edits only.
- The course template repository, devcontainer, and Colab notebooks working and tested.
- Project materials, rubrics, and Canvas setup complete.

Deferred to summer 2027 or later: the LLM "predict, then explain" feature, additional videos beyond one per module, new Build It labs, polish, and anything flagged *Deferred* in [DEVELOPMENT_BACKLOG.md](DEVELOPMENT_BACKLOG.md).

## 14. Target course architecture

Two repositories. The existing course repository keeps its Quarto publishing setup and holds everything students read; the new reference system template holds everything students run. This is proposal P-12 in the decision log.

### Course repository (`uc-bana-7075`)

```
uc-bana-7075/
├── README.md                    course overview, objectives, Week 0–7 table
├── GLOSSARY.md                  course design vocabulary (not published)
├── planning/                    these planning documents (not published)
├── docs/adr/                    architecture decision records (not published)
├── book/                        Quarto book, published to GitHub Pages
│   ├── _quarto.yml              parts named by week
│   ├── 00-orientation.qmd       Week 0 (new)
│   ├── 01–14 chapters           Learn It, See It, Try It, Apply It, Build It sections
│   ├── appendix-*.qmd           ML primer, continued learning, serving methods
│   ├── _templates/              module page template
│   └── styles/                  Human Side and Build with AI callout styles
├── slides/                      one deck per week
├── labs/                        Build It code: apple-demand ModelOps, YouTube pipeline
├── papers/                      case study readings
└── project/                     charter, design document, and role templates; milestone instructions
```

Kept private (gitignored, as today): `grading/`, syllabi, `recordings/`, lab check answer keys, and the instructor guide unless Q-06 decides otherwise.

### Reference system template (new repository)

```
bana7075-reference-system/       working name (P-10)
├── .devcontainer/               Codespaces and local VS Code
├── .github/workflows/           tests; scheduled fresh-install check
├── data/                        down-sampled or synthetic sample data
├── artifacts/                   precomputed outputs for each stage
├── scenarios/                   packaged failure scenarios
├── src/                         data, features, models, monitoring
├── deployment/                  FastAPI service, Streamlit app
├── notebooks/try-it/            Colab notebooks for Try Its
├── notebooks/evidence/          Colab templates for each role's evidence minimum
├── tests/
└── pyproject.toml, uv.lock      pinned dependencies
```

Canvas holds lab checks, the Week 0 checklist, peer evaluations, and grades.

## 15. Changes from `revamp_spec.md`

| Spec section | Change |
|---|---|
| §3, §7 Week 3 | Explainability removed from the Week 3 title and focus; code and callouts only |
| §6 | Reference system decided as simplified Rossmann; YouTube pipeline optional, may be replaced |
| §8 | Explainability row removed; Git competency done in the web interface in the team repository |
| §9 | Colab for notebooks, Codespaces for servers, local VS Code self-service |
| §10 | Build with AI is a callout, not a module section; AI-use policy is a Week 0 pre-read; LLM feature deferred |
| §11 | Tiers replaced by role-owned component evidence with evidence minimums; teams pick their own problem and dataset; reliability lead added for five-person teams |
| §11 milestones | Final defense is the team design document plus a recorded presentation with one graded segment per member |
| §12 | Weights changed from 40/25/20/15 to 30/20/25/20/5, making 70% individual |
