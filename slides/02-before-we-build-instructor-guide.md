---
title: "Instructor Guide: Discovery Phase Exercise"
format:
  pdf:
    toc: false
    number-sections: true
    colorlinks: true
---

**Purpose:** This guide provides detailed questions and prompts to facilitate deeper discussion during student presentations and help identify gaps in their discovery thinking.

---

## Area 1: Stakeholder Engagement

### Key Questions Students Should Consider

**Core Questions:**

- Who would the key stakeholders be for this project?
- What questions would you ask these stakeholders to ensure you understand the business problem?
- What assumptions might be uncovered during these discussions?

**Detailed Sub-Questions:**

- Who are the key stakeholders? (executives, department leads, data science teams, IT, end users)
- What questions would help you understand:
  - The real business problem and objectives?
  - Current processes and their limitations?
  - Success criteria and constraints?
  - Data availability and access?
- What assumptions might stakeholders have that you need to validate?
- Who will be impacted by this solution?

### Prompting Questions During Presentations

- "You mentioned talking to [stakeholder X], but what about [stakeholder Y]? Why might their perspective matter?"
- "What assumptions is [stakeholder type] likely making that could derail the project?"
- "How would you prioritize which stakeholders to talk to first? Why?"
- "What do you think you DON'T know that could be a major blocker?"
- "If stakeholders give you conflicting requirements, how would you resolve that?"

### Common Gaps to Watch For

- ✗ Only identifying business stakeholders, missing technical/IT stakeholders
- ✗ Not thinking about end users who will interact with the system
- ✗ Not considering who will maintain the system long-term
- ✗ Missing data governance or compliance stakeholders (especially in regulated industries)
- ✗ Not validating assumptions about data availability or quality

### Great Answers to Highlight

- ✓ Identifies both technical and business stakeholders
- ✓ Recognizes potential conflicts between stakeholder priorities
- ✓ Plans to validate specific assumptions (e.g., "We're assuming churn data exists, but we need to verify")
- ✓ Considers who will be negatively impacted and how to engage them

---

## Area 2: Evaluating ML Suitability

### Key Questions Students Should Consider

**Core Questions:**

- Do you think ML is a suitable solution? What factors would you consider to determine if ML is appropriate for this problem?
- Provide an example of an alternative, non-ML approach that could be considered. What are the limitations of this approach compared to an ML approach?

**Detailed Considerations:**

- Does this problem require learning complex patterns from data?
- Will the system need to make predictions on new, unseen data?
- Is quality data available (or obtainable)?
- Could a simpler solution work? (rule-based systems, heuristics, traditional analytics)
- What are the trade-offs between ML and non-ML approaches?
- Are there interpretability or regulatory requirements?

### Prompting Questions During Presentations

- "You said ML is appropriate—but could you solve this with a simple rule-based system? Why or why not?"
- "What would make you decide NOT to use ML for this problem?"
- "How would you test if ML is actually needed before committing to building it?"
- "What's the cost of being wrong? (false positives vs. false negatives)"
- "If stakeholders need to understand every decision, how does that affect your ML approach?"
- "What happens if you can't get enough quality data? What's your plan B?"

### Common Gaps to Watch For

- ✗ Assuming ML is always the answer without considering alternatives
- ✗ Not thinking about interpretability requirements (especially for fraud, hiring, healthcare)
- ✗ Ignoring data availability and quality concerns
- ✗ Not considering maintenance complexity vs. simpler solutions
- ✗ Missing regulatory or compliance constraints

### Great Answers to Highlight

- ✓ Acknowledges when a non-ML solution might be sufficient
- ✓ Identifies specific trade-offs (e.g., "ML gives better accuracy but requires ongoing retraining")
- ✓ Considers interpretability needs (e.g., "For hiring, we need to explain decisions to avoid bias concerns")
- ✓ Plans to validate ML is needed before full commitment (e.g., "Run a simple baseline first")

---

## Area 3: Performance Metrics

### Key Questions Students Should Consider

**Core Questions:**

- Define three performance metrics for the ML system. Include at least:
  - **One technical metric** (e.g., model performance like RMSE, precision, recall)
  - **One system performance metric** (e.g., response time, uptime, scalability)
  - **One business metric** (e.g., customer retention, fraud reduction, cost savings)
- Explain why each of these metrics is important for evaluating the success of the ML system.

**Detailed Considerations:**

- **Model performance metrics**: Accuracy, precision, recall, RMSE, etc. - which matter for this problem?
- **System performance metrics**: Latency, throughput, uptime, scalability - what are the operational requirements?
- **Business metrics**: Revenue impact, cost savings, customer satisfaction, efficiency gains - how does this drive business value?
- How do these metrics align with stakeholder priorities?
- What trade-offs exist between different metrics?

### Prompting Questions During Presentations

- "You mentioned [metric X]—who cares about that metric? Business leaders, engineers, or end users?"
- "What trade-offs exist between your model metric and your business metric?"
- "If you had to pick ONE metric that matters most to stakeholders, what would it be? Why?"
- "How would you handle a situation where your model accuracy is high but business value is low?"
- "For real-time systems (like fraud detection), what happens if latency is too high?"
- "How would you set the threshold for 'good enough' on each metric?"

### Common Gaps to Watch For

- ✗ Only listing model metrics (accuracy, precision) without business or system metrics
- ✗ Not connecting metrics to stakeholder priorities
- ✗ Missing critical system metrics (especially latency for real-time systems)
- ✗ Not considering metric trade-offs (e.g., precision vs. recall in fraud detection)
- ✗ Picking metrics that don't align with the actual business goal

### Great Answers to Highlight

- ✓ Balances all three metric types (model, system, business)
- ✓ Explains WHY each metric matters and WHO cares about it
- ✓ Identifies trade-offs (e.g., "High recall means catching more fraud but more false positives")
- ✓ Connects metrics to business outcomes (e.g., "Reducing churn by 5% = $X million saved")

---

## Area 4: Understanding Value and Feasibility

### Key Questions Students Should Consider

**Core Questions:**

- What's the potential value of the ML system to the organization? Consider both:
  - **Tangible benefits** (e.g., cost savings, revenue increase, efficiency improvements)
  - **Intangible benefits** (e.g., improved customer satisfaction, brand reputation)
- List some key **technical requirements** that would be helpful to understand early on before developing the solution (e.g., data, infrastructure).
- What gaps might exist, and how would you address them?

**Detailed Considerations:**

**Value:**

- Tangible benefits: cost savings, revenue increase, time savings?
- Intangible benefits: customer satisfaction, brand reputation, employee morale?
- Does the potential value justify the investment?

**Feasibility:**

- Data requirements: What data exists? What's missing? Quality issues?
- Infrastructure: Compute resources, deployment environment, integration needs?
- Algorithm constraints: Real-time vs. batch? Interpretability needs?
- Team capabilities: Skills, tools, timeline?

### Prompting Questions During Presentations

- "You estimated $X in savings—how did you arrive at that number? What assumptions are you making?"
- "What if the data you need doesn't exist or is poor quality? How does that change the value proposition?"
- "If this costs $Y to build and maintain, is the value still worth it?"
- "What gaps exist between what you need and what you have? (data, skills, infrastructure)"
- "How would you de-risk this project early? What could you validate quickly?"
- "What's the MVP version of this solution? Could you prove value with something simpler first?"

### Common Gaps to Watch For

- ✗ Vague value statements ("improve customer satisfaction") without quantification
- ✗ Not considering total cost of ownership (development + maintenance + infrastructure)
- ✗ Assuming all necessary data exists and is high quality
- ✗ Missing infrastructure requirements (e.g., real-time systems need different architecture)
- ✗ Not identifying skill gaps on the team
- ✗ No plan for addressing feasibility gaps

### Great Answers to Highlight

- ✓ Quantifies both tangible and intangible benefits with reasonable assumptions
- ✓ Identifies specific data, infrastructure, and skill gaps
- ✓ Proposes concrete ways to address gaps (acquire data, upskill team, use cloud)
- ✓ Suggests de-risking approaches (MVP, proof of concept)
- ✓ Considers ROI: does value justify cost?

---

## Area 5: Iterative Development Process

### Key Questions Students Should Consider

**Core Questions:**

- Describe why the development of this ML system would be an **iterative process**.
- Provide an example of something that could **change during development** (e.g., a performance metric, a technical requirement).
- How would you **manage this change** to avoid scope creep?

**Detailed Considerations:**

- What might change as you learn more? (metrics, requirements, scope, data needs)
- Why can't you define everything perfectly upfront?
- How would you structure the work? (MVP first? Phased rollout?)
- How do you avoid scope creep while remaining flexible?
- What feedback loops need to exist? (stakeholders, users, model performance)
- When would you revisit initial assumptions and decisions?

### Prompting Questions During Presentations

- "You said [X] might change—how would you know when to revisit that decision?"
- "What feedback loops would you establish to catch problems early?"
- "How do you balance being flexible vs. avoiding scope creep?"
- "If stakeholders keep adding requirements, how would you push back?"
- "What would trigger you to stop and reassess the entire approach?"
- "How would you structure releases? Big bang or incremental rollout?"

### Common Gaps to Watch For

- ✗ Not recognizing that requirements will evolve as you learn more
- ✗ No plan for managing changing requirements
- ✗ Missing feedback loops (stakeholder check-ins, model monitoring)
- ✗ No strategy for avoiding scope creep
- ✗ Assuming everything can be defined perfectly upfront

### Great Answers to Highlight

- ✓ Recognizes specific things that will likely change (metrics, data needs, scope)
- ✓ Proposes concrete iteration structure (MVP → pilot → full rollout)
- ✓ Plans for feedback loops (weekly stakeholder check-ins, A/B testing)
- ✓ Has strategy for scope management (fixed milestones, change approval process)
- ✓ Understands that discovery itself is iterative

---

## Cross-Cutting Discussion Questions

Use these to facilitate discussion **after all presentations**:

### Comparing Approaches

- "Group X approached stakeholder engagement by [approach], while Group Y did [different approach]. What are the pros/cons of each?"
- "Which group identified a concern that others missed? Why might that be important?"
- "Did anyone propose NOT using ML? Why or why not?"

### Connecting to Reality

- "How does this exercise connect to the Booking.com paper you'll read this week?"
- "In your own work experience, have you seen examples of projects that failed because discovery was insufficient?"
- "What surprised you most about this exercise?"

### Forward-Looking

- "How will you use this discovery thinking in your course project proposal?"
- "What's the ONE thing you'll take away from today's exercise?"
