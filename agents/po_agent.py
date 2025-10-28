"""
Product Owner Agent - Generates Product Requirements Document (PRD)
Synthesizes all analysis into actionable requirements document
"""

import dspy

from signatures.dspy_signatures import PRDGenerator


class POAgent:
    """
    Generates comprehensive PRD by synthesizing:
    - ML recommendations (use case, features, models)
    - Deployment strategy (infrastructure, team, costs)
    - Business communication (ROI, stakeholders)
    - Quality assessment (risks, data issues)

    Output: Structured PRD ready for stakeholder review
    """

    def __init__(self):
        self.generator = dspy.ChainOfThought(PRDGenerator)

    def generate_prd(
        self,
        schema_results: dict,
        quality_results: dict,
        ml_recommendations: dict,
        deployment_strategy: dict,
        business_communication: dict,
    ) -> dict:
        """
        Generate PRD from all agent outputs

        Args:
            schema_results: Output from SchemaAgent
            quality_results: Output from QualityAgent
            ml_recommendations: Output from MLAdvisorAgent
            deployment_strategy: Output from DeploymentAgent
            business_communication: Output from BusinessCommunicationAgent

        Returns:
            Dict with PRD document
        """
        # Prepare synthesized inputs
        ml_use_case = self._format_ml_use_case(ml_recommendations)
        feature_engineering = self._format_feature_engineering(ml_recommendations)
        deployment_summary = self._format_deployment_summary(deployment_strategy)
        business_summary = self._format_business_summary(business_communication)
        quality_issues = self._format_quality_issues(quality_results)

        try:
            prd_output = self.generator(
                ml_use_case=ml_use_case,
                feature_engineering=feature_engineering,
                deployment_strategy=deployment_summary,
                business_summary=business_summary,
                quality_issues=quality_issues,
            )

            return {"prd_document": prd_output.prd_document, "status": "success"}

        except Exception as e:
            print(f"❌ PRD generation failed: {str(e)}")
            return {
                "prd_document": f"# PRD Generation Failed\n\nError: {str(e)}",
                "status": "error",
            }

    def _format_ml_use_case(self, ml_recommendations: dict) -> str:
        """Format ML use case summary"""
        use_case = ml_recommendations["ml_use_case"]
        return f"""
**Use Case**: {use_case['detected_use_case']}
**Target Variable**: {use_case['target_variable']}
**ML Readiness**: {use_case['suitability_score']}/100
**Reasoning**: {use_case['target_reasoning']}
**Alternative**: {use_case.get('alternative_use_case', 'N/A')}
"""

    def _format_feature_engineering(self, ml_recommendations: dict) -> str:
        """Format feature engineering summary"""
        feature_eng = ml_recommendations["feature_engineering"]
        return f"""
**Feature Plan**:
{feature_eng.get('feature_plan', 'N/A')[:500]}...

**Training Strategy**:
{feature_eng.get('training_recommendations', 'N/A')}

**MLflow Setup**:
{feature_eng.get('mlflow_setup', 'N/A')}
"""

    def _format_deployment_summary(self, deployment_strategy: dict) -> str:
        """Format deployment summary"""
        return f"""
**Team Requirements**: {deployment_strategy.get('team_requirements', 'N/A')[:300]}...
**Timeline**: {deployment_strategy.get('implementation_roadmap', 'N/A')[:300]}...
**Costs**: {deployment_strategy.get('cost_estimation', 'N/A')[:300]}...
**Infrastructure**: {deployment_strategy.get('databricks_setup', 'N/A')[:300]}...
**Monitoring**: {deployment_strategy.get('monitoring_plan', 'N/A')[:200]}...
"""

    def _format_business_summary(self, business_communication: dict) -> str:
        """Format business summary"""
        return f"""
**Executive Summary**: {business_communication.get('executive_summary', 'N/A')[:400]}...
**ROI Justification**: {business_communication.get('budget_justification', 'N/A')[:300]}...
**Success Metrics**: {business_communication.get('stakeholder_talking_points', 'N/A')[:200]}...
"""

    def _format_quality_issues(self, quality_results: dict) -> str:
        """Format quality issues summary"""
        summary = quality_results.get("summary", {})
        return f"""
**Total Issues**: {summary.get('total_issues', 0)}
**Critical**: {summary.get('critical', 0)}
**Warnings**: {summary.get('warnings', 0)}
**Info**: {summary.get('info', 0)}
"""
