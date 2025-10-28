"""
DSPy Signatures - Programmatic Prompt Definitions
Each signature defines input/output structure for LLM reasoning
"""

import dspy


class SchemaInterpreter(dspy.Signature):
    """
    Interprets pandas data type and infers business meaning of a column.
    Uses sample values and statistics to make informed inference.
    """

    # Inputs (programmatically computed)
    column_name = dspy.InputField(desc="Name of the column")
    pandas_dtype = dspy.InputField(
        desc="Pandas data type (e.g., int64, object, float64)"
    )
    null_count = dspy.InputField(desc="Number of null values")
    unique_count = dspy.InputField(desc="Number of unique values")
    total_count = dspy.InputField(desc="Total number of rows")
    sample_values = dspy.InputField(desc="List of 2 sample non-null values as string")

    # Outputs (LLM reasoning)
    business_type = dspy.OutputField(
        desc="Business type: 'Identifier', 'Categorical', 'Numeric Metric', 'Date/Time', 'Text', or 'Boolean'"
    )
    confidence = dspy.OutputField(desc="Confidence level: 'high', 'medium', or 'low'")
    reasoning = dspy.OutputField(desc="One sentence explanation for the classification")
    recommendation = dspy.OutputField(
        desc="Brief recommendation: 'Keep', 'Review', or 'Consider dropping' with reason"
    )


class StatisticalInsightGenerator(dspy.Signature):
    """
    Generate business insights from statistical summary of a column.
    Interprets patterns and provides actionable recommendations.
    """

    # Inputs (programmatically computed)
    column_name = dspy.InputField(desc="Name of the column")
    column_type = dspy.InputField(desc="Type: 'numeric' or 'categorical'")
    stats_dict = dspy.InputField(
        desc="Dictionary with statistics (mean/median/std for numeric, cardinality/top_values for categorical)"
    )

    # Outputs (LLM reasoning)
    insight = dspy.OutputField(
        desc="1 sentence business insight explaining what the statistics reveal"
    )
    pattern_detected = dspy.OutputField(
        desc="For numeric: 'normal', 'right skewed', 'left skewed', 'bimodal', 'uniform'. For categorical: 'high cardinality', 'low cardinality', 'binary', 'dominant category'"
    )
    actionable_suggestion = dspy.OutputField(
        desc="Specific action analyst should take based on the pattern (e.g., 'investigate outliers', 'consider grouping rare categories')"
    )


class QualityRecommender(dspy.Signature):
    """
    Recommend actions to fix data quality issues.
    Provides actionable code snippets and impact assessment.
    """

    issue_type = dspy.InputField(
        desc="Type: 'missing_values', 'duplicates', 'outliers', 'inconsistent_categories'"
    )
    column_name = dspy.InputField(desc="Affected column name")
    issue_description = dspy.InputField(
        desc="Description of the quality issue with counts/percentages"
    )
    sample_data = dspy.InputField(desc="Sample of affected data")

    recommended_action = dspy.OutputField(
        desc="Specific recommendation (e.g., 'Impute with median', 'Drop duplicates', 'Cap outliers')"
    )
    python_code = dspy.OutputField(desc="Pandas code snippet to fix the issue")
    impact_description = dspy.OutputField(
        desc="What will change after applying this fix"
    )


class MLUseCaseDetector(dspy.Signature):
    """
    Detects suitable ML use case and target variable from dataset analysis.
    Provides reasoning and ML readiness assessment.
    """

    dataset_overview = dspy.InputField(
        desc="Dataset overview: row count, column count, column types summary"
    )
    key_columns = dspy.InputField(
        desc="List of important columns with their types and characteristics"
    )
    quality_issues = dspy.InputField(desc="Summary of data quality problems found")

    detected_use_case = dspy.OutputField(
        desc="Primary ML use case: regression, classification, clustering, or time-series"
    )
    target_variable = dspy.OutputField(desc="Recommended target column name")
    target_reasoning = dspy.OutputField(
        desc="2-sentence explanation why this target makes sense"
    )
    suitability_score = dspy.OutputField(desc="ML readiness score 0-100")
    alternative_use_case = dspy.OutputField(
        desc="Alternative ML approach if applicable"
    )


class FeatureEngineeringPlanner(dspy.Signature):
    """
    Generates feature engineering recommendations in markdown format.
    Provides column-by-column transformation strategy.
    """

    column_summary = dspy.InputField(
        desc="Key columns with types, cardinality, null%, patterns"
    )
    target_variable = dspy.InputField(desc="Selected target variable")
    ml_use_case = dspy.InputField(desc="Selected ML use case")
    planning_instructions = dspy.InputField(
        desc="Use case-specific instructions for planning depth and focus areas"
    )

    feature_plan = dspy.OutputField(
        desc="Markdown formatted feature engineering plan with transformations per column"
    )
    training_recommendations = dspy.OutputField(
        desc="Model suggestions, validation strategy, hyperparameter hints in 3-4 sentences"
    )
    mlflow_setup = dspy.OutputField(
        desc="MLflow experiment tracking recommendations in 2-3 sentences"
    )


class DatabricksDeploymentPlanner(dspy.Signature):
    """
    Generates comprehensive MLOps deployment strategy covering technical, organizational, and operational aspects.
    Provides end-to-end roadmap from development to production.
    """

    ml_use_case = dspy.InputField(desc="Detected ML use case and target variable")
    feature_plan = dspy.InputField(desc="Feature engineering strategy")
    training_plan = dspy.InputField(desc="Model training recommendations")
    data_summary = dspy.InputField(desc="Dataset schema and quality summary")

    # Technical Infrastructure
    databricks_setup = dspy.OutputField(
        desc="Unity Catalog structure, cluster configurations, MLflow experiment setup in markdown with ## headers"
    )
    serving_strategy = dspy.OutputField(
        desc="Model serving endpoint configuration, API design, versioning strategy, scaling considerations in markdown"
    )
    monitoring_plan = dspy.OutputField(
        desc="Data drift detection, model performance tracking, alerting setup, dashboard recommendations in markdown"
    )
    data_strategy = dspy.OutputField(
        desc="Data pipeline architecture, refresh frequency, retention policies, backup strategy in markdown"
    )

    # Team & Timeline
    team_requirements = dspy.OutputField(
        desc="Required roles (data engineers, data scientists, MLOps, architect), FTE estimates, skill requirements, ramp-up timeline in markdown"
    )
    implementation_roadmap = dspy.OutputField(
        desc="Phase-wise timeline in weeks (POC: X weeks, Development: Y weeks, UAT: Z weeks, Production: W weeks) with key milestones and deliverables in markdown"
    )
    risk_mitigation = dspy.OutputField(
        desc="Technical risks, organizational dependencies, data quality risks, mitigation strategies with ownership in markdown"
    )

    # Governance & Business
    cost_estimation = dspy.OutputField(
        desc="Databricks compute costs, storage costs, serving endpoint costs, monthly estimates, optimization strategies in markdown"
    )
    governance_framework = dspy.OutputField(
        desc="Unity Catalog permissions, model approval workflow, data access controls, compliance requirements (GDPR/SOC2) in markdown"
    )
    success_metrics = dspy.OutputField(
        desc="Business KPIs to track, model performance metrics, operational SLAs, reporting cadence in markdown"
    )
    business_impact = dspy.OutputField(
        desc="ROI estimation, business value drivers, efficiency gains, stakeholder communication plan in markdown"
    )

    # Operations & Quality
    testing_framework = dspy.OutputField(
        desc="Unit testing strategy, integration tests, model validation tests, data quality tests, CI/CD pipeline in markdown"
    )
    operational_playbook = dspy.OutputField(
        desc="Incident response procedures, model degradation handling, data pipeline failure recovery, rollback strategy in markdown"
    )
    enablement_plan = dspy.OutputField(
        desc="Documentation requirements, training sessions for stakeholders, runbooks for operations, knowledge transfer checklist in markdown"
    )

    # Future Vision
    future_enhancements = dspy.OutputField(
        desc="Feature store adoption roadmap, A/B testing framework, AutoML integration, model marketplace strategy, advanced monitoring in markdown"
    )


class BusinessCommunicationGenerator(dspy.Signature):
    """
    Generates executive-ready business communication materials.
    Translates technical ML strategy into stakeholder-friendly formats.
    """

    ml_use_case = dspy.InputField(
        desc="ML use case, target variable, and readiness score"
    )
    deployment_summary = dspy.InputField(
        desc="Key highlights from deployment strategy: team size, timeline, costs"
    )
    technical_risks = dspy.InputField(
        desc="Summary of technical and organizational risks"
    )
    success_metrics = dspy.InputField(
        desc="Business KPIs and model performance metrics"
    )

    executive_summary = dspy.OutputField(
        desc="1-page executive summary in plain English: problem, solution, value, investment, timeline. Use markdown headers and bullet points."
    )
    risk_matrix = dspy.OutputField(
        desc="Risk prioritization matrix in markdown table format with Impact (High/Medium/Low) × Likelihood (High/Medium/Low) grid"
    )
    timeline_visual = dspy.OutputField(
        desc="Mermaid Gantt chart syntax for project timeline with phases: POC, Development, UAT, Production"
    )
    budget_justification = dspy.OutputField(
        desc="Cost breakdown with ROI projection in markdown: investment vs expected returns with payback period"
    )
    stakeholder_talking_points = dspy.OutputField(
        desc="Key messages for different audiences: executives, technical teams, finance, operations in markdown with ## headers"
    )


class PRDGenerator(dspy.Signature):
    """
    Generates production-grade Product Requirements Document (PRD).
    Follows industry best practices for ML product specifications.
    """

    ml_use_case = dspy.InputField(desc="ML use case, target variable, and suitability")
    feature_engineering = dspy.InputField(
        desc="Feature engineering plan and training strategy"
    )
    deployment_strategy = dspy.InputField(
        desc="Technical infrastructure, team, timeline, and costs"
    )
    business_summary = dspy.InputField(
        desc="Executive summary, ROI, and stakeholder communication"
    )
    quality_issues = dspy.InputField(desc="Data quality summary and risks")

    prd_document = dspy.OutputField(
        desc="""Generate a comprehensive, production-ready PRD in markdown:

# Product Requirements Document (PRD)
**Project:** [Auto-generate from ML use case]
**Owner:** [Product Manager - TBD]
**Status:** Draft
**Last Updated:** [Current date]

---

## 1. Executive Summary (1 paragraph)
What we're building, why, expected impact, and ask (budget/resources).

## 2. Problem Statement
### 2.1 Current State & Pain Points
- Quantified business problem (e.g., "$2M annual cost, 15% error rate")
- Target user personas with demographics and current workflows
- Why existing solutions fail

### 2.2 Market & Competitive Context
- Competitive landscape: What do alternatives offer?
- Why now? (Regulatory change, market shift, tech enablement)

### 2.3 Product Vision (North Star)
12-24 month aspirational goal for this ML capability.

## 3. Goals & Success Metrics
### 3.1 Business Objectives
| Metric | Owner | Baseline | Target | Timeline |
|--------|-------|----------|--------|----------|
| [Revenue impact] | VP Sales | $5M | $7M | Q3 2025 |
| [Cost reduction] | CFO | $2M | $1.2M | Q4 2025 |

### 3.2 Product/ML Metrics
- Model performance: Accuracy, precision, recall with thresholds
- Operational SLAs: Latency, uptime, error rates
- Adoption metrics: DAU, API calls, user satisfaction

### 3.3 Leading Indicators
What we'll track weekly/monthly to predict success.

## 4. User Personas & Journeys
For each key persona (2-4):
- **[Persona Name]**: Role, goals, pain points, tech-savviness
- **Current workflow**: How they solve this today (step-by-step)
- **Future workflow**: How ML changes their experience

## 5. Functional Requirements

### 5.1 User Stories (Prioritized)
Format each as:
**[Priority: P0/P1/P2] Story ID: Title**
**Epic:** [Group related stories]
**Effort:** [Story points]
**Dependencies:** [Blocking stories/tickets]

As a [specific role], I want to [action] so that [measurable outcome].

**Acceptance Criteria:**
- [Functional: What the system does]
- [Technical: Performance, scalability thresholds]
- [Data Quality: Input validation, error handling]
- [Security: Auth, encryption, compliance]

**Out of Scope:** [What this story explicitly doesn't cover]

### 5.2 Non-Functional Requirements
- **Performance**: Latency, throughput, concurrency limits
- **Scalability**: Current vs. future capacity needs
- **Security**: Authentication, authorization, encryption, auditing
- **Compliance**: GDPR, HIPAA, SOC2, industry-specific regulations
- **Accessibility**: WCAG 2.1 AA standards (if user-facing)

## 6. Technical Architecture

### 6.1 System Context Diagram
```
[Upstream Data Sources] → [Data Pipeline] → [Feature Store] → [Model Training] → [Model Registry] → [Prediction API] → [Downstream Consumers]
                                ↓
                          [Monitoring & Alerting]
```

### 6.2 API Contracts
- **Endpoint:** POST /api/v1/predict
- **Request Schema:** `{customer_id: str, features: {...}}`
- **Response Schema:** `{prediction: float, confidence: float, model_version: str}`
- **Error Codes:** 400 (bad input), 429 (rate limit), 500 (model error)

### 6.3 Data Flow & Storage
- Input data sources and refresh cadence
- Feature engineering pipeline (batch/streaming)
- Model artifacts storage and versioning
- Prediction results retention policy

### 6.4 Failure Modes & Resilience
- What happens when model service is down? (Fallback to rules, cached predictions)
- Data pipeline failures: Alerts, auto-retry, manual intervention
- Model degradation: Auto-rollback criteria

## 7. ML-Specific Considerations

### 7.1 Model Lifecycle Management
- **Training:** Cadence (weekly/trigger-based), compute requirements
- **Validation:** Holdout set, cross-validation, A/B testing strategy
- **Deployment:** Blue-green, canary, shadow mode rollout
- **Monitoring:** Drift detection, performance tracking, alerting thresholds
- **Retraining:** Automated triggers (drift >10%, accuracy <80%)

### 7.2 Explainability & Transparency
- Feature importance visualization for data scientists
- Prediction explanations for end-users (SHAP, LIME)
- Model cards documenting training data, limitations, bias audits

### 7.3 Bias & Fairness
- Protected attributes identified (race, gender, age)
- Fairness metrics: Demographic parity, equalized odds
- Audit cadence and remediation process

## 8. Risk Assessment & Mitigation

| Risk | Impact | Likelihood | Owner | Mitigation | Trigger |
|------|--------|------------|-------|------------|---------|
| [Data quality degrades] | High | Medium | Data Eng | Automated validation, alerts | >5% null rate |
| [Model drift] | Critical | Low | ML Lead | Weekly monitoring, auto-retrain | Accuracy <80% for 2 weeks |
| [Regulatory audit fails] | Critical | Low | Compliance | Pre-launch audit, quarterly reviews | N/A |

## 9. Implementation Roadmap

### Phase 0: POC (Weeks 1-4)
- **Goal:** Prove technical feasibility
- **Deliverables:** Jupyter notebook, accuracy >75% on sample data
- **Success Criteria:** Stakeholder demo approval

### Phase 1: MVP (Weeks 5-12)
- **Goal:** Ship to 10% of users in shadow mode
- **Deliverables:** Prediction API, monitoring dashboard
- **Success Criteria:** 99% uptime, <200ms latency

### Phase 2: Scale (Weeks 13-20)
- **Goal:** 100% rollout, automated retraining
- **Deliverables:** Production pipeline, on-call runbook
- **Success Criteria:** Business KPIs met (see Section 3)

### Phase 3: Optimize (Weeks 21+)
- **Goal:** Cost optimization, feature improvements
- **Backlog:** Real-time predictions, multi-model ensembles

## 10. Operating Model (Post-Launch)

### 10.1 Roles & Responsibilities
- **Model Owner:** [ML Lead] - Model performance, retraining decisions
- **API Owner:** [Backend Lead] - Uptime, latency, integration support
- **Data Owner:** [Data Eng] - Pipeline reliability, data quality

### 10.2 On-Call & Incident Response
- **Severity 1** (Model down): Page ML on-call, rollback to last-known-good
- **Severity 2** (Degraded accuracy): Investigate within 4 hours, retrain if needed
- **Runbook:** Link to internal wiki with troubleshooting steps

### 10.3 Continuous Improvement
- Weekly model performance review
- Monthly backlog grooming: User feedback → prioritized features
- Quarterly architecture review: Scalability, tech debt

## 11. Go-to-Market & Adoption

### 11.1 Rollout Strategy
- **Week 1-2:** Internal beta (data science team)
- **Week 3-4:** Shadow mode (predictions logged, not acted on)
- **Week 5-6:** 10% traffic canary
- **Week 7-8:** 50% rollout
- **Week 9+:** 100% rollout

### 11.2 Rollback Criteria
Auto-rollback if:
- Error rate >5% for 10 minutes
- Latency p95 >500ms for 5 minutes
- User complaints >10/hour

### 11.3 User Onboarding
- Training sessions: 2-hour workshop for end-users
- Documentation: API docs, user guides, FAQs
- Support channels: Slack #ml-support, tickets

### 11.4 Change Management
- Communicate value: "Saves 2 hours/day per analyst"
- Address concerns: "Model provides suggestions, you make final call"
- Gather feedback: Weekly office hours, quarterly surveys

## 12. Compliance & Legal

### 12.1 Data Privacy
- GDPR: Right to explanation, data portability, deletion
- CCPA: Opt-out mechanism, data inventory
- Data retention: Training data 90 days, predictions 1 year

### 12.2 Regulatory Approvals
- [Industry-specific]: FDA (medical devices), OCC (banking), etc.
- Timeline: 8-12 weeks for submission and approval

### 12.3 Audit Trail
- All predictions logged with model version, timestamp, input features
- Access logs for compliance audits

## 13. Budget & Resources

### 13.1 Team Requirements
- 2 Data Scientists (model development, tuning)
- 1.5 Data Engineers (pipeline, infrastructure)
- 0.5 ML Engineer (deployment, monitoring)
- 0.25 Product Manager (roadmap, stakeholder management)

### 13.2 Cost Breakdown (Monthly)
- **Development:** $X salaries, $Y cloud sandbox
- **Production Run Rate:** $A compute, $B storage, $C API costs
- **Tooling:** $D MLflow, $E monitoring (Datadog), $F experiment tracking

### 13.3 ROI Projection
- **Investment:** $500K (Year 1)
- **Expected Return:** $1.2M annual savings
- **Payback Period:** 5 months

## 14. Dependencies & Assumptions

### 14.1 Critical Dependencies
- **Data Access:** Legal approval for customer PII (ETA: Week 2)
- **Infrastructure:** Databricks workspace provisioned (ETA: Week 1)
- **Integrations:** CRM API access granted (ETA: Week 4)

### 14.2 Key Assumptions
- Data quality remains stable (validated monthly)
- Model retraining compute budget approved
- No major regulatory changes in next 12 months

## 15. Out of Scope (Explicit Non-Goals)

- Real-time streaming predictions (<50ms latency)
- Multi-language support (English only for MVP)
- Mobile app (web dashboard only)
- [Add others based on stakeholder expectations]

## 16. Open Questions & Decisions Needed

- [ ] **Decision:** Model refresh cadence - weekly or trigger-based? (Owner: ML Lead, Due: Week 2)
- [ ] **Question:** Should we support batch predictions for historical analysis? (Owner: Product, Due: Week 3)
- [ ] **Blocker:** Legal review of GDPR compliance incomplete (Owner: Legal, Due: Week 4)

## 17. Appendix

### 17.1 Glossary
- **AUC-ROC:** Area Under Receiver Operating Characteristic curve (model performance metric)
- **Drift:** Statistical change in input data distribution over time
- **Feature Store:** Centralized repository for ML features

### 17.2 References
- [Link to data schema documentation]
- [Link to model experiment results]
- [Link to competitive analysis]

### 17.3 Revision History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-01-15 | AI System | Initial draft |

---

**CRITICAL INSTRUCTIONS:**
1. Be SPECIFIC to the detected ML use case - avoid generic boilerplate
2. Use real numbers from the analysis (row counts, column types, quality issues)
3. Generate 4-6 user stories covering end-user, data scientist, and ops personas
4. Include measurable acceptance criteria with thresholds
5. Reference actual deployment costs and timelines from previous agents
6. Highlight data quality risks found in quality analysis
7. Use markdown tables for metrics, risks, and roadmap
8. Keep executive summary under 150 words
9. Ensure all sections are actionable - no "TBD" without owner and deadline
10. Format for readability: Headers, bullets, tables, code blocks where appropriate"""
    )
