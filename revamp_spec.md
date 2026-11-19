# BANA 7075: ML System Design for Business — Course Redesign

## 1. Background and motivation

BANA 7075 is a seven-week graduate course at the University of Cincinnati focused on designing, developing, deploying, and maintaining end-to-end machine learning systems.

The course serves students with varying technical backgrounds, including M.S. Business Analytics, MBA, and AI Certificate students. Some students aspire to develop ML systems, while others will manage technical teams, lead AI initiatives, or make decisions about ML systems.

Recent student feedback suggests several opportunities for improvement:

- Provide more guided exercises, demonstrations, and opportunities to practice before students are expected to build independently.
- Reduce friction associated with environment setup, package installation, Git, and command-line tools.
- Improve accessibility for students with limited programming experience without sacrificing technical rigor.
- Preserve the end-to-end system development experience that technically proficient students have identified as particularly valuable.
- Improve group project structure, collaboration, and individual accountability.
- Incorporate AI-assisted development practices.

The goal is not to make the course less technical. The goal is to make technical content more accessible and provide opportunities for differentiated implementation depth.

## 2. Agreed-upon instructional philosophy

The redesigned course will follow three guiding principles:

1. **Common conceptual core:** All students learn the fundamental concepts, architectural patterns, engineering practices, and business considerations associated with ML systems.
2. **Common hands-on minimum:** All students interact with real ML systems, execute provided workflows, inspect results, modify configurations, diagnose problems, and evaluate system behavior.
3. **Optional deeper engineering:** Students seeking greater technical depth can develop, extend, and independently implement system components.

We do not want separate technical and nontechnical courses or rigid student tracks. Instead, we want a shared instructional experience with progressively greater implementation depth.

The learning progression should be:

Understand → Run → Observe → Explain → Modify → Build

Understanding, operating, evaluating, troubleshooting, and guided experimentation are required. Independent software development is optional.

## 3. Course identity

The redesigned course should develop the technical literacy and practical competencies needed to design, evaluate, operate, and improve end-to-end machine learning systems.

Students should learn through real system interactions, guided engineering exercises, architectural decisions, and collaborative projects.

The course should balance the needs of students who will build ML systems and students who will lead, manage, or evaluate those systems.

BANA 7075 should remain focused primarily on predictive ML systems and their supporting infrastructure. A separate proposed Applied AI Engineering course will focus more extensively on LLM applications, RAG, agents, and AI-powered application development.

## 4. Proposed learning objectives

Upon completing the course, all students should be able to:

1. **Understand:** Explain the components, dependencies, and lifecycle of an end-to-end ML system.
2. **Design:** Design ML system architectures that align with business requirements and technical constraints.
3. **Operate:** Execute and interact with data pipelines, modeling workflows, inference services, and monitoring tools.
4. **Evaluate:** Assess system performance, reliability, scalability, maintainability, and business value.
5. **Troubleshoot:** Diagnose common system failures and recommend appropriate corrective actions.
6. **Collaborate:** Communicate and defend technical decisions within a cross-functional ML development team.

An additional objective for students pursuing greater technical depth is to implement and extend ML systems using modern software engineering and MLOps practices.

## 5. Standard instructional framework

Every weekly module and corresponding textbook section should follow a consistent instructional framework.

### Learn It — Required

Introduce concepts, terminology, architectural patterns, business implications, and engineering tradeoffs.

### See It — Required

Demonstrate how concepts appear in a real ML system through short walkthrough videos, architecture diagrams, screenshots, and working examples.

### Try It — Required

Provide guided, low-friction technical activities that allow students to:

- Execute existing code.
- Interact with ML applications.
- Modify configurations or parameters.
- Inspect system artifacts and outputs.
- Introduce controlled failures.
- Diagnose and explain system behavior.

These activities should not require students to write substantial amounts of Python from scratch.

### Apply It — Required

Have students apply their understanding through:

- System design decisions.
- Industry case studies.
- Architecture reviews.
- System teardown exercises.
- Technical debt assessments.
- Team project contributions.

### Build It — Optional

Provide deeper implementation exercises for students interested in developing their engineering skills.

These activities can involve implementing pipelines, writing tests, developing APIs, automating workflows, and extending reference architectures.

Existing technical notebooks and exercises should be reused where appropriate.

## 6. Reference system strategy

The course should ideally revolve around one canonical reference ML system that students interact with throughout all seven weeks.

The reference system should demonstrate:

- Business problem framing and requirements.
- Data ingestion and transformation.
- Data validation.
- Model training and evaluation.
- Experiment tracking.
- Model registration and versioning.
- Model explainability.
- Batch and/or real-time inference.
- Application integration.
- Automated testing.
- CI/CD.
- Model and data monitoring.
- Retraining and maintenance considerations.

Students should see this system early in the course and progressively explore its individual components.

The reference system should include intentional failure scenarios, such as invalid input data, deteriorating model performance, failed automated tests, drifted data, and problematic architectural decisions.

Supplement the primary reference system with two or three contrasting industry case studies or reference architectures. These do not necessarily need to be fully implemented systems.

Potential contrasts include:

- Batch versus real-time inference.
- Classification versus forecasting or recommendation.
- Small-scale versus enterprise ML infrastructure.
- Predictive ML versus systems incorporating generative AI.

### Reference system decision still open

Potential approaches include:

- Reuse and modernize the existing YouTube ML system.
- Develop a new retail demand forecasting system.
- Adapt an existing Rossmann forecasting application.
- Build another business-focused reference application.

Evaluate these options based on instructional value, maintenance burden, accessibility, existing reusable assets, and coverage of the course learning objectives.

The sophistication of the predictive model is secondary. The system surrounding the model is the primary subject of the course.

Do not assume that a new reference system must be built before evaluating the existing assets.

## 7. Proposed seven-week curriculum

| Week | Topic | Central design question |
|---|---|---|
| 1 | ML system architecture and foundations | What are we building, and what business outcome does it support? |
| 2 | DataOps | How do we ensure trustworthy data flows into our system? |
| 3 | ModelOps and explainability | How do we develop, compare, reproduce, and explain models? |
| 4 | Model serving and deployment | How should predictions be delivered to users and downstream systems? |
| 5 | DevOps and system reliability | How do we safely introduce changes without breaking the system? |
| 6 | Monitoring and operations | How do we know when a deployed ML system stops performing effectively? |
| 7 | System integration and defense | Can we demonstrate and defend the complete system? |

This is a proposed curriculum structure, not a finalized sequence. Evaluate it against existing course materials and learning objectives.

## 8. Minimum practical competencies

Every student should individually demonstrate the following competencies.

| Domain | Required competency | Optional deeper engineering |
|---|---|---|
| Architecture | Trace data and predictions through a working system | Implement a system component |
| Git and GitHub | Create an issue, edit a file, and submit a pull request | Manage complex branches and automated workflows |
| DataOps | Run a pipeline, introduce invalid data, and diagnose a failure | Develop pipelines and custom validation checks |
| ModelOps | Run experiments, compare results, and justify model selection | Develop training and registration workflows |
| Explainability | Interpret model explanations and individual predictions | Implement explainability functionality |
| Deployment | Submit inference requests and evaluate responses | Build or extend inference APIs |
| DevOps | Inspect automated tests and diagnose workflow failures | Write tests and implement CI/CD |
| Monitoring | Interpret drift and performance reports and recommend actions | Implement monitors and retraining triggers |

Required technical activities should be scaffolded sufficiently for students with limited programming experience.

## 9. Technology and infrastructure strategy

Minimize the infrastructure and tooling friction students encounter.

Potential technologies include:

- Python
- Google Colab
- GitHub and GitHub web interface
- GitHub Codespaces
- GitHub Actions
- scikit-learn
- MLflow
- FastAPI
- Streamlit
- Evidently
- pytest

These technologies are candidates, not mandatory architectural decisions.

Guiding principles:

- Required activities should involve minimal or no local environment setup.
- Prefer browser-based experiences.
- Introduce GitHub progressively throughout the course.
- Make command-line development optional where possible.
- Pin dependencies and provide reproducible environments.
- Provide fallback instructional artifacts when hosted services are unavailable.
- Minimize reliance on multiple third-party hosting services.
- Verify current free-tier limitations before selecting hosting platforms.
- Reuse existing course assets whenever practical.

Week 0 should include a readiness assessment, environment orientation, and supplemental Python and machine learning resources.

## 10. AI integration

AI should be incorporated in two distinct ways.

### AI as part of the ML system

Explore opportunities to enhance traditional predictive ML systems with AI capabilities.

One potential example is a "predict, then explain" feature in which an LLM generates accessible explanations grounded in actual model predictions and explainability outputs.

This should remain secondary to the core predictive ML systems curriculum. LLM development, RAG, and agent engineering should not dominate the course.

### AI-assisted engineering

Encourage students to use AI coding assistants to:

- Understand unfamiliar code.
- Explain errors.
- Generate and modify code.
- Develop tests.
- Review architectural decisions.
- Produce documentation.
- Troubleshoot system failures.

Students remain responsible for understanding, testing, and verifying AI-generated outputs.

Each module should include a short Build with AI section describing a relevant use case, example, and potential risk.

Assessments should reward evidence of verification rather than the amount of manually written code.

## 11. Proposed final project redesign

Retain a collaborative, team-based project but shift the emphasis toward system design, decision-making, and demonstrable practical evidence.

All teams should produce a common ML system design document covering:

- Business problem and stakeholders.
- Functional and nonfunctional requirements.
- Success metrics.
- System architecture.
- Data sources and pipeline design.
- Data quality and validation.
- Model development and evaluation strategy.
- Model explainability.
- Deployment and inference architecture.
- Testing and automation.
- Monitoring and retraining.
- Security, risks, and responsible AI.
- Technical tradeoffs and limitations.

### Proposed implementation tiers

Teams may choose an implementation depth appropriate to their experience.

**Tier 1: Analyze and adapt**

Produce a complete system design and demonstrate controlled experiments or configuration changes using the reference system.

**Tier 2: Extend**

Produce a complete system design and implement meaningful extensions to the reference system.

**Tier 3: Engineer**

Produce a complete system design and develop a substantially independent ML system.

Tier 2 should be encouraged when feasible.

All tiers should be capable of earning the same maximum grade. Evaluation should focus on the quality of decisions, explanations, practical evidence, verification, and architectural reasoning rather than the amount of code written.

### Proposed team structure

Teams should ideally contain four students with complementary backgrounds.

Potential roles include:

- Product and business lead.
- Data lead.
- ML lead.
- Platform and operations lead.

Use a Week 0 survey to understand student backgrounds, experience, availability, and role preferences.

The instructor should facilitate team formation.

Every student should contribute to design decisions and practical verification. Avoid creating teams where one technical student is responsible for all implementation.

Use individual activities, individually attributable project contributions, reflections, and peer evaluations to establish accountability.

### Proposed project milestones

- Week 1: Team formation and charter.
- Week 2: Business problem, requirements, and initial architecture.
- Week 4: Data, model, and experiment design.
- Week 6: Deployment, testing, and monitoring design.
- Week 7: Final design document, practical implementation evidence, and presentation.

## 12. Proposed assessment strategy

Consider the following preliminary grading structure:

- Team system design and project milestones: 40%.
- Individual guided technical activities: 25%.
- Individual system teardown and architectural analysis assignments: 20%.
- Individual final defense and reflection: 15%.

These percentages are preliminary and should be evaluated during course development.

The overarching goal is to balance collaborative learning with meaningful assessment of individual competency.

## 13. Existing course content and assets

Before developing new content, conduct a comprehensive inventory of the existing course repository.

Identify:

- Textbook chapters and concepts.
- Existing notebooks.
- Data engineering pipelines.
- Model development workflows.
- MLflow examples.
- FastAPI and Streamlit applications.
- Monitoring and drift reports.
- Git and CI/CD exercises.
- Existing assignments and quizzes.
- Case studies and readings.
- Project instructions and rubrics.
- Course syllabus and learning objectives.
- Website infrastructure and publishing workflows.

Determine which assets should be:

- Retained.
- Reorganized.
- Simplified.
- Modernized.
- Extended.
- Converted into guided activities.
- Moved into optional Build It sections.
- Replaced or retired.

Avoid unnecessary redevelopment of existing high-quality content.

# Requested development planning tasks

Act as a senior ML engineer, curriculum architect, and technical project manager helping redesign this course.

Do not immediately begin implementing the redesigned course. First develop a comprehensive, repository-informed development plan.

## Task 1: Audit the existing repository

Explore the repository structure, source files, instructional materials, technical examples, dependencies, and publishing infrastructure.

Document:

- Current course architecture.
- Existing instructional and technical assets.
- Opportunities for reuse.
- Gaps relative to the redesigned course.
- Technical debt and maintenance concerns.
- Dependencies between existing components.

Support findings with actual file paths and repository evidence rather than assumptions.

## Task 2: Perform a curriculum gap analysis

Map existing material to:

- Proposed weekly modules.
- Revised learning objectives.
- Common practical competencies.
- Learn It, See It, Try It, Apply It, and Build It activities.

Identify missing content and distinguish content needing minor revisions from content requiring substantial redevelopment.

## Task 3: Evaluate reference system options

Evaluate whether the existing reference system can support the redesigned curriculum or whether a new or adapted system would be preferable.

Provide a decision matrix and recommendation.

Consider:

- Business relevance.
- Instructional coverage.
- Technical complexity.
- Student accessibility.
- Reusability.
- Hosting requirements.
- Maintenance cost.
- Ability to demonstrate controlled failures.

## Task 4: Define the target course architecture

Propose the future organization of:

- Course textbook.
- Weekly instructional modules.
- Guided technical activities.
- Optional engineering labs.
- Reference system source code.
- Case studies.
- Datasets and example artifacts.
- Assignments and assessments.
- Team project materials.
- Instructor resources.
- Technical infrastructure.

Recommend an appropriate repository organization based on the existing publishing framework.

## Task 5: Create a development backlog

Break the redesign into epics, tasks, and actionable TODOs.

For each task, include:

- Unique task identifier.
- Description.
- Rationale.
- Priority.
- Dependencies.
- Estimated effort: small, medium, or large.
- Deliverables.
- Acceptance criteria.
- Existing files affected, where identifiable.
- New files or components required, where identifiable.

Organize the backlog into logical epics, such as:

1. Course identity and curriculum design.
2. Repository and content restructuring.
3. Reference ML system.
4. Student development environments.
5. Weekly instructional content.
6. Guided practical activities.
7. Optional engineering labs.
8. AI-assisted development integration.
9. Case studies and system teardowns.
10. Team project redesign.
11. Assessment and grading.
12. Documentation and instructional videos.
13. Testing and course quality assurance.

Distinguish tasks required for the initial course launch from enhancements that can be deferred.

## Task 6: Develop a phased implementation roadmap

Recommend a development sequence that minimizes unnecessary rework.

Consider beginning with:

- Course repository audit.
- Finalization of instructional requirements.
- Reference system selection.
- Development of one complete pilot module.
- Validation of the instructional approach.
- Development of remaining modules.
- Assessment and project development.
- End-to-end course testing.

Identify critical decisions that need instructor approval before dependent development begins.

## Task 7: Create planning documents

Create or propose the following Markdown files in an appropriate planning directory:

- `COURSE_REDESIGN.md`: Course vision, instructional philosophy, requirements, and proposed curriculum.
- `CURRENT_STATE_AUDIT.md`: Inventory and evaluation of the existing course.
- `CURRICULUM_MAP.md`: Weekly modules, objectives, activities, assessments, and existing content mappings.
- `REFERENCE_SYSTEM_OPTIONS.md`: Evaluation of reference system alternatives.
- `DEVELOPMENT_BACKLOG.md`: Epics, actionable tasks, dependencies, and acceptance criteria.
- `IMPLEMENTATION_ROADMAP.md`: Recommended phases and development sequence.
- `DECISION_LOG.md`: Confirmed decisions, proposed decisions, and unresolved questions.

Use repository conventions where appropriate rather than introducing redundant planning infrastructure.

## Development constraints

- Preserve valuable existing instructional content.
- Do not reduce academic or technical rigor simply to accommodate different programming backgrounds.
- Make required activities accessible without extensive environment setup.
- Preserve substantial engineering opportunities for advanced students.
- Prioritize maintainability and reproducibility.
- Avoid unnecessary technology proliferation.
- Keep the scope realistic for a seven-week accelerated graduate course.
- Clearly distinguish confirmed decisions from recommendations and assumptions.
- Do not make major architectural or curriculum changes before presenting the development plan for review.

The initial deliverable should be a comprehensive, actionable development plan that can guide subsequent Claude Code implementation sessions.