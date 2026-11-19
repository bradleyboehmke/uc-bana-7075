# BANA 7075 Course Design

The shared language for designing and redesigning BANA 7075 (ML System Design for Business): its instructional units, example systems, and project structure.

## Language

### Course structure

**Week**:
One of the course's eight time periods, Week 0 (readiness and orientation) through Week 7.
_Avoid_: Session, unit

**Module**:
The instructional unit for one week, made up of Learn It, See It, Try It, Apply It, and Build It.
_Avoid_: Lesson, unit, topic

**Chapter**:
One section of the course textbook; a module can draw on several chapters.
_Avoid_: Reading, lesson

### Module components

**Learn It**:
The required part of a module that introduces concepts, terminology, patterns, and tradeoffs.
_Avoid_: Lecture, reading

**See It**:
The required part of a module that demonstrates its concepts in a working system.
_Avoid_: Demo, walkthrough

**Try It**:
The required, guided activity in which each student runs, inspects, modifies, or breaks a working system without writing substantial code.
_Avoid_: Lab, exercise, hands-on

**Lab check**:
The individually graded Canvas quiz that confirms a student completed a Try It and understood what they observed.
_Avoid_: Lab report, Try It quiz

**Apply It**:
The required part of a module in which students use its concepts to make or critique design decisions.
_Avoid_: Assignment, discussion

**Build It**:
The optional, ungraded implementation work in a module for students who want deeper engineering practice.
_Avoid_: Extra credit, advanced track, bonus lab

**Human Side callout**:
A recurring chapter callout that raises the people, organizational, and ethical considerations of the topic at hand.
_Avoid_: Human in the loop callout, ethics box

**Build with AI callout**:
A recurring chapter callout that shows how an AI coding assistant can help with the topic at hand, with an example and a risk to verify.
_Avoid_: AI section, AI tip

**Human in the loop**:
A point within an ML system's workflow where a person reviews, approves, or corrects what the system does.
_Avoid_: Human side

### Example systems

**Reference system**:
The single end-to-end ML system that students follow through every week of the course, serving as the model of good architecture and implementation.
_Avoid_: Canonical system, course system, demo system

**Supplementary example**:
An additional system or artifact demonstrated within one module to show that module's topic in a second setting, alongside the reference system.
_Avoid_: Secondary system, side example

**Case study**:
A real organization's ML system that students read about and analyze rather than run.
_Avoid_: Supplementary example, example system

### Team project

**Team system**:
The ML system a project team designs for a business problem and dataset of its own choosing.
_Avoid_: Final project system, student system

**Design document**:
The team's written deliverable that describes and justifies the team system's design.
_Avoid_: Technical report, final report, design doc

**Component evidence**:
The working practical results a team member produces on the team's own data for the part of the team system their role owns.
_Avoid_: Implementation, deliverable, tier work

**Evidence minimum**:
The required component evidence for a role, which done excellently can earn full marks.
_Avoid_: Tier 1, baseline deliverable

**Deeper evidence**:
Optional component evidence beyond the evidence minimum, recognized in feedback but not awarded extra points.
_Avoid_: Tier, extra credit, Build It

**Contribution adjustment**:
An instructor's discretionary reduction of a student's share of the team grade, informed by peer evaluations.
_Avoid_: Peer grade, peer evaluation score

### Team roles

**Product lead**:
The team member who owns the business problem, success metrics, and risk and responsible AI review.
_Avoid_: Business lead, project manager

**Data lead**:
The team member who owns data sources, data quality, and validation.
_Avoid_: Data engineer

**ML lead**:
The team member who owns experiments, model selection, and model registration.
_Avoid_: Data scientist, modeler

**Platform lead**:
The team member who owns serving predictions and monitoring them.
_Avoid_: Ops lead, MLOps engineer

**Reliability lead**:
The fifth team member, present only in five-person teams, who owns tests and automated workflows.
_Avoid_: QA lead, DevOps engineer
