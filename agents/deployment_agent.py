"""
Deployment Agent - Generates comprehensive MLOps strategy
Provides team sizing, timelines, costs, governance, and operational plans
"""
import dspy
from signatures.dspy_signatures import DatabricksDeploymentPlanner


class DeploymentAgent:
    """
    Generates end-to-end deployment strategy covering:
    - Technical infrastructure (Databricks, MLflow, serving)
    - Team requirements & resource planning
    - Phase-wise implementation roadmap with timelines
    - Cost estimation & optimization
    - Governance, testing, and operations
    - Business impact & future enhancements
    """
    
    def __init__(self):
        self.planner = dspy.ChainOfThought(DatabricksDeploymentPlanner)
    
    def analyze(self, schema_results: dict, ml_recommendations: dict) -> dict:
        """
        Generate comprehensive deployment strategy
        
        Args:
            schema_results: Output from SchemaAgent
            ml_recommendations: Output from MLAdvisorAgent
            
        Returns:
            Dict with complete deployment strategy
        """
        # Prepare inputs
        ml_use_case = self._format_ml_use_case(ml_recommendations)
        feature_plan = ml_recommendations['feature_engineering']['feature_plan']
        training_plan = ml_recommendations['feature_engineering']['training_recommendations']
        data_summary = self._format_data_summary(schema_results)
        
        try:
            deployment_plan = self.planner(
                ml_use_case=ml_use_case,
                feature_plan=feature_plan,
                training_plan=training_plan,
                data_summary=data_summary
            )
            
            return {
                # Technical Infrastructure
                'databricks_setup': deployment_plan.databricks_setup,
                'serving_strategy': deployment_plan.serving_strategy,
                'monitoring_plan': deployment_plan.monitoring_plan,
                'data_strategy': deployment_plan.data_strategy,
                
                # Team & Timeline
                'team_requirements': deployment_plan.team_requirements,
                'implementation_roadmap': deployment_plan.implementation_roadmap,
                'risk_mitigation': deployment_plan.risk_mitigation,
                
                # Governance & Business
                'cost_estimation': deployment_plan.cost_estimation,
                'governance_framework': deployment_plan.governance_framework,
                'success_metrics': deployment_plan.success_metrics,
                'business_impact': deployment_plan.business_impact,
                
                # Operations & Quality
                'testing_framework': deployment_plan.testing_framework,
                'operational_playbook': deployment_plan.operational_playbook,
                'enablement_plan': deployment_plan.enablement_plan,
                
                # Future Vision
                'future_enhancements': deployment_plan.future_enhancements
            }
            
        except Exception as e:
            print(f"❌ Deployment planning failed: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _format_ml_use_case(self, ml_recommendations: dict) -> str:
        """Format ML use case for deployment planning"""
        use_case = ml_recommendations['ml_use_case']
        return f"{use_case['detected_use_case']} | Target: {use_case['target_variable']} | Score: {use_case['suitability_score']}/100"
    
    def _format_data_summary(self, schema_results: dict) -> str:
        """Format data summary with key metrics"""
        summary = schema_results['summary']
        return f"{summary['total_rows']} rows, {summary['total_columns']} columns, {summary['memory_usage_mb']:.1f}MB"
    
    def _generate_error_response(self, error: str) -> dict:
        """Generate structured error response"""
        error_msg = f"Error: {error}"
        return {
            'databricks_setup': error_msg,
            'serving_strategy': "Unable to generate",
            'monitoring_plan': "Unable to generate",
            'data_strategy': "Unable to generate",
            'team_requirements': "Unable to generate",
            'implementation_roadmap': "Unable to generate",
            'risk_mitigation': "Unable to generate",
            'cost_estimation': "Unable to generate",
            'governance_framework': "Unable to generate",
            'success_metrics': "Unable to generate",
            'business_impact': "Unable to generate",
            'testing_framework': "Unable to generate",
            'operational_playbook': "Unable to generate",
            'enablement_plan': "Unable to generate",
            'future_enhancements': "Unable to generate"
        }