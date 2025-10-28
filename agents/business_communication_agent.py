"""
Business Communication Agent - Generates executive-ready materials
Translates technical strategy into stakeholder-friendly formats
"""
import dspy
from signatures.dspy_signatures import BusinessCommunicationGenerator


class BusinessCommunicationAgent:
    """
    Generates business communication materials:
    - Executive summary (1-pager)
    - Risk prioritization matrix
    - Visual timeline (Mermaid)
    - Budget justification with ROI
    - Stakeholder-specific talking points
    """
    
    def __init__(self):
        self.generator = dspy.ChainOfThought(BusinessCommunicationGenerator)
    
    def analyze(self, ml_recommendations: dict, deployment_strategy: dict) -> dict:
        """
        Generate business communication materials
        
        Args:
            ml_recommendations: Output from MLAdvisorAgent
            deployment_strategy: Output from DeploymentAgent
            
        Returns:
            Dict with executive-ready materials
        """
        # Prepare inputs
        ml_use_case = self._format_ml_use_case(ml_recommendations)
        deployment_summary = self._format_deployment_summary(deployment_strategy)
        technical_risks = deployment_strategy.get('risk_mitigation', 'No risks identified')
        success_metrics = deployment_strategy.get('success_metrics', 'No metrics defined')
        
        try:
            communication_materials = self.generator(
                ml_use_case=ml_use_case,
                deployment_summary=deployment_summary,
                technical_risks=technical_risks,
                success_metrics=success_metrics
            )
            
            return {
                'executive_summary': communication_materials.executive_summary,
                'risk_matrix': communication_materials.risk_matrix,
                'timeline_visual': communication_materials.timeline_visual,
                'budget_justification': communication_materials.budget_justification,
                'stakeholder_talking_points': communication_materials.stakeholder_talking_points
            }
            
        except Exception as e:
            print(f"❌ Business communication generation failed: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _format_ml_use_case(self, ml_recommendations: dict) -> str:
        """Format ML use case summary"""
        use_case = ml_recommendations['ml_use_case']
        return f"{use_case['detected_use_case']} targeting {use_case['target_variable']} (Readiness: {use_case['suitability_score']}/100)"
    
    def _format_deployment_summary(self, deployment_strategy: dict) -> str:
        """Format deployment highlights"""
        team = deployment_strategy.get('team_requirements', 'Team size not estimated')
        timeline = deployment_strategy.get('implementation_roadmap', 'Timeline not defined')
        costs = deployment_strategy.get('cost_estimation', 'Costs not estimated')
        
        # Extract key numbers if available (simplified)
        summary = f"Team: {team[:100]}... | Timeline: {timeline[:100]}... | Costs: {costs[:100]}..."
        return summary
    
    def _generate_error_response(self, error: str) -> dict:
        """Generate error response"""
        error_msg = f"Error: {error}"
        return {
            'executive_summary': error_msg,
            'risk_matrix': "Unable to generate",
            'timeline_visual': "Unable to generate",
            'budget_justification': "Unable to generate",
            'stakeholder_talking_points': "Unable to generate"
        }