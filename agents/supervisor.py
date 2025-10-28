"""
Supervisor Agent - Orchestrates workflow
Coordinates Schema, Profile, Quality, ML Advisor, and Deployment agents
"""
import pandas as pd
import dspy
from agents.schema_agent import SchemaAgent
from agents.profile_agent import ProfileAgent
from agents.quality_agent import QualityAgent
from agents.ml_advisor_agent import MLAdvisorAgent
from agents.deployment_agent import DeploymentAgent
from agents.business_communication_agent import BusinessCommunicationAgent
from agents.po_agent import POAgent
from config import OPENAI_API_KEY, OPENAI_MODEL


class SupervisorAgent:
    """
    Orchestrates the data analysis workflow.
    Coordinates: Schema Agent, Profile Agent, Quality Agent, ML Advisor Agent, Deployment Agent
    """
    
    def __init__(self):
        """Initialize supervisor and configure DSPy with OpenAI"""
        # Configure DSPy to use OpenAI
        from dspy_init import get_configured_lm
        get_configured_lm()
        
        # Initialize all agents
        self.schema_agent = SchemaAgent()
        self.profile_agent = ProfileAgent()
        self.quality_agent = QualityAgent()
        self.ml_advisor_agent = MLAdvisorAgent()
        self.deployment_agent = DeploymentAgent()
        self.business_communication_agent = BusinessCommunicationAgent()
        self.po_agent = POAgent()
    
    def analyze_dataset(self, df: pd.DataFrame) -> dict:
        """
        Coordinate analysis workflow on uploaded dataset
        
        Args:
            df: pandas DataFrame to analyze
            
        Returns:
            dict with analysis results from all agents
        """
        results = {
            'status': 'in_progress',
            'agents_completed': [],
            'schema_analysis': None,
            'profile_analysis': None,
            'quality_analysis': None,
            'ml_recommendations': None,
            'deployment_strategy': None,
            'business_communication': None,
            'errors': []
        }
        
        # Step 1: Schema Analysis
        try:
            print("🔍 Running Schema Agent...")
            schema_results = self.schema_agent.analyze(df)
            results['schema_analysis'] = schema_results
            results['agents_completed'].append('schema_agent')
            print("✅ Schema Agent completed")
        except Exception as e:
            error_msg = f"Schema Agent failed: {str(e)}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        # Step 2: Statistical Profiling
        try:
            print("📊 Running Profile Agent...")
            profile_results = self.profile_agent.analyze(df)
            results['profile_analysis'] = profile_results
            results['agents_completed'].append('profile_agent')
            print("✅ Profile Agent completed")
        except Exception as e:
            error_msg = f"Profile Agent failed: {str(e)}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        # Step 3: Quality Checks
        try:
            print("🔍 Running Quality Agent...")
            quality_results = self.quality_agent.analyze(df)
            results['quality_analysis'] = quality_results
            results['agents_completed'].append('quality_agent')
            print("✅ Quality Agent completed")
        except Exception as e:
            error_msg = f"Quality Agent failed: {str(e)}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        # Step 4: ML Advisor (synthesizes all previous results)
        if results['schema_analysis'] and results['profile_analysis'] and results['quality_analysis']:
            try:
                print("🤖 Running ML Advisor Agent...")
                ml_recommendations = self.ml_advisor_agent.analyze(
                    schema_results=results['schema_analysis'],
                    profile_results=results['profile_analysis'],
                    quality_results=results['quality_analysis']
                )
                results['ml_recommendations'] = ml_recommendations
                results['agents_completed'].append('ml_advisor_agent')
                print("✅ ML Advisor Agent completed")
            except Exception as e:
                error_msg = f"ML Advisor Agent failed: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Step 5: Deployment Strategy
        if results['ml_recommendations']:
            try:
                print("🚀 Running Deployment Agent...")
                deployment_strategy = self.deployment_agent.analyze(
                    schema_results=results['schema_analysis'],
                    ml_recommendations=results['ml_recommendations']
                )
                results['deployment_strategy'] = deployment_strategy
                results['agents_completed'].append('deployment_agent')
                print("✅ Deployment Agent completed")
            except Exception as e:
                error_msg = f"Deployment Agent failed: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")

        # Step 6: Business Communication Materials
        if results['ml_recommendations'] and results['deployment_strategy']:
            try:
                print("📊 Running Business Communication Agent...")
                business_materials = self.business_communication_agent.analyze(
                    ml_recommendations=results['ml_recommendations'],
                    deployment_strategy=results['deployment_strategy']
                )
                results['business_communication'] = business_materials
                results['agents_completed'].append('business_communication_agent')
                print("✅ Business Communication Agent completed")
            except Exception as e:
                error_msg = f"Business Communication Agent failed: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Update status
        if len(results['errors']) == 0:
            results['status'] = 'completed'
        else:
            results['status'] = 'partial_failure'
        
        return results
    
    def get_summary(self, results: dict) -> str:
        """
        Generate human-readable summary of analysis
        
        Args:
            results: Output from analyze_dataset()
            
        Returns:
            Formatted summary string
        """
        if results['status'] == 'completed':
            schema = results['schema_analysis']
            quality = results.get('quality_analysis', {})
            summary = f"""
            ✅ Analysis Complete!
            
            Dataset Overview:
            - Total Rows: {schema['summary']['total_rows']:,}
            - Total Columns: {schema['summary']['total_columns']}
            - Memory Usage: {schema['summary']['memory_usage_mb']:.2f} MB
            
            Quality Summary:
            - Issues Found: {quality.get('summary', {}).get('total_issues', 0)}
            - Critical: {quality.get('summary', {}).get('critical', 0)}
            - Warnings: {quality.get('summary', {}).get('warnings', 0)}
            
            Agents Completed: {', '.join(results['agents_completed'])}
            """
            return summary
        else:
            return f"⚠️ Analysis completed with errors: {', '.join(results['errors'])}"
        

    # Add method for PRD generation (separate from analyze_dataset)
    def generate_prd(self, results: dict) -> dict:
        """
        Generate PRD from existing analysis results
        Does NOT re-run agents - uses cached results
        
        Args:
            results: Previously generated results from analyze_dataset()
            
        Returns:
            Dict with PRD document
        """
        if not all([
            results.get('schema_analysis'),
            results.get('quality_analysis'),
            results.get('ml_recommendations'),
            results.get('deployment_strategy'),
            results.get('business_communication')
        ]):
            return {
                'prd_document': "# Error\n\nInsufficient data to generate PRD. Please complete all analysis steps first.",
                'status': 'error'
            }
        
        try:
            print("📝 Generating PRD...")
            prd_result = self.po_agent.generate_prd(
                schema_results=results['schema_analysis'],
                quality_results=results['quality_analysis'],
                ml_recommendations=results['ml_recommendations'],
                deployment_strategy=results['deployment_strategy'],
                business_communication=results['business_communication']
            )
            print("✅ PRD generated successfully")
            return prd_result
        except Exception as e:
            error_msg = f"PRD generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'prd_document': f"# Error\n\n{error_msg}",
                'status': 'error'
            }