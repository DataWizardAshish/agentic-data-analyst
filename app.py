"""
Streamlit App - Phase 1A: Schema Analysis Only
Single-page interface for uploading CSV and viewing schema analysis
"""
import streamlit as st
import pandas as pd
import dspy
from agents.supervisor import SupervisorAgent
import sys
import os
from config import OPENAI_API_KEY, OPENAI_MODEL

st.cache_data.clear()
st.cache_resource.clear()

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# if "dspy_initialized" not in st.session_state:
#     dspy.configure(lm=dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY))
#     st.session_state.dspy_initialized = True
# from dspy_init import configure_dspy
# configure_dspy()

# Page configuration
st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'prd_result' not in st.session_state:
    st.session_state.prd_result = None
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

supervisor = SupervisorAgent()
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    
    
    st.title("🤖 Intelligent Agentic ML Product Planner")
    st.markdown("---")
    
    # STEP 1: Upload Data
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze its schema"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            st.session_state.analysis_results = None
            st.session_state.prd_result = None
        
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file, encoding="latin-1")
            st.session_state.uploaded_df = df
            
            # Show preview
            st.success(f"✅ File loaded: {uploaded_file.name}")
            with st.expander("📊 Data Preview (First 5 rows)"):
                st.dataframe(df.head())
            
            st.markdown("---")
            
            # STEP 2: Analyze Button
            st.header("⚙️ MLops Agents")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button(
                    "🔍 Get ML Product Strategy",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_button:
                with st.spinner("🤖 Agents are analyzing your data..."):
                    # Initialize supervisor and run analysis
                    #supervisor = SupervisorAgent()
                    results = supervisor.analyze_dataset(df)
                    #st.write("DEBUG: Analysis results keys ->", list(results.keys()))
                    st.session_state.analysis_results = results
                
                st.success("✅ Analysis Complete!")
            
            # STEP 3: Display Results
            if st.session_state.analysis_results is not None:
                #st.write("DEBUG: Loaded analysis results from session_state")
                st.markdown("---")
                st.header("📊 Schema Analysis")
                
                results = st.session_state.analysis_results
                #st.write("DEBUG: Results sections available ->", list(results.keys()))
                
                # Summary metrics
                if results['schema_analysis']:
                    schema = results['schema_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Rows", f"{schema['summary']['total_rows']:,}")
                    col2.metric("Total Columns", schema['summary']['total_columns'])
                    col3.metric("Memory Usage", f"{schema['summary']['memory_usage_mb']:.2f} MB")
                    
                    st.markdown("### 🔍 Column-by-Column Analysis")
                    
                    # Create detailed table
                    columns_data = schema['columns']
                    
                    for col_data in columns_data:
                        with st.expander(f"**{col_data['column_name']}** - {col_data['business_type']} ({col_data['confidence']} confidence)"):
                            
                            col_left, col_right = st.columns(2)
                            
                            with col_left:
                                st.markdown("**📋 Technical Details**")
                                st.write(f"- **Pandas Type**: `{col_data['pandas_dtype']}`")
                                st.write(f"- **Null Count**: {col_data['null_count']} ({col_data['null_percentage']}%)")
                                st.write(f"- **Unique Values**: {col_data['unique_count']}")
                                st.write(f"- **Sample Values**: {col_data['sample_values']}")
                            
                            with col_right:
                                st.markdown("**🤖 AI Interpretation**")
                                st.write(f"- **Business Type**: {col_data['business_type']}")
                                st.write(f"- **Confidence**: {col_data['confidence']}")
                                st.write(f"- **Reasoning**: {col_data['reasoning']}")
                                
                                # Color-code recommendation
                                rec = col_data['recommendation']
                                if 'Keep' in rec:
                                    st.success(f"✅ {rec}")
                                elif 'Review' in rec:
                                    st.warning(f"⚠️ {rec}")
                                else:
                                    st.info(f"💡 {rec}")
                    
                    # Summary table view
                    st.markdown("### 📋 Summary Table View")
                    summary_df = pd.DataFrame([
                        {
                            'Column': c['column_name'],
                            'Type': c['pandas_dtype'],
                            'Business Type': c['business_type'],
                            'Nulls %': f"{c['null_percentage']}%",
                            'Unique': c['unique_count'],
                            'Confidence': c['confidence'],
                            'Recommendation': c['recommendation']
                        }
                        for c in columns_data
                    ])
                    st.dataframe(summary_df, use_container_width=True)
                
                # Profile Analysis Section
                if results.get('profile_analysis'):
                    st.markdown("---")
                    st.markdown("### 📊 Statistical Profiling")
                    
                    profile = results['profile_analysis']
                    
                    # Numeric columns
                    if profile.get('numeric_analysis') and len(profile['numeric_analysis']) > 0:
                        st.markdown("#### 🔢 Numeric Columns")
                        for col_data in profile['numeric_analysis']:
                            with st.expander(f"**{col_data['column_name']}** - {col_data['pattern_detected']}"):
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    st.markdown("**📈 Statistics**")
                                    st.write(f"- **Mean**: {col_data['mean']:.2f}")
                                    st.write(f"- **Median**: {col_data['median']:.2f}")
                                    st.write(f"- **Std Dev**: {col_data['std']:.2f}")
                                    st.write(f"- **Min**: {col_data['min']:.2f}")
                                    st.write(f"- **Max**: {col_data['max']:.2f}")
                                    st.write(f"- **25th Percentile**: {col_data['q25']:.2f}")
                                    st.write(f"- **75th Percentile**: {col_data['q75']:.2f}")
                                    st.write(f"- **Skewness**: {col_data['skewness']:.2f}")
                                
                                with col_right:
                                    st.markdown("**💡 AI Insights**")
                                    st.info(col_data['insight'])
                                    st.warning(f"🔍 **Suggestion**: {col_data['actionable_suggestion']}")
                
                # Quality Analysis Section
                if results.get('quality_analysis'):
                    st.markdown("---")
                    st.markdown("### ⚠️ Data Quality Issues")
                    
                    quality = results['quality_analysis']
                    summary = quality['summary']
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Issues", summary['total_issues'])
                    col2.metric("Critical", summary.get('critical', 0))
                    col3.metric("Warnings", summary.get('warnings', 0))
                    col4.metric("Info", summary.get('info', 0))
                    
                    if summary['total_issues'] == 0:
                        st.success("✅ No data quality issues detected!")
                    else:
                        st.markdown("#### 🔍 Detected Issues")
                        
                        for issue in quality['issues_found']:
                            severity = issue['severity']
                            icon = "🔴" if severity == 'critical' else "⚠️" if severity == 'warnings' else "ℹ️"
                            
                            with st.expander(f"{icon} {issue['type'].replace('_', ' ').title()} - {issue['column']} ({issue.get('count', 0)} issues)"):
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    st.markdown("**📋 Issue Details**")
                                    st.write(f"- **Type**: {issue['type']}")
                                    st.write(f"- **Column**: {issue['column']}")
                                    st.write(f"- **Severity**: {severity}")
                                    st.write(f"- **Description**: {issue['description']}")
                                    if 'percentage' in issue:
                                        st.write(f"- **Affected**: {issue['percentage']:.2f}% of data")
                                
                                with col_right:
                                    st.markdown("**💡 Recommended Fix**")
                                    st.info(f"**Action**: {issue['recommended_action']}")
                                    st.code(issue['code_snippet'], language='python')
                                    st.success(f"**Impact**: {issue['impact']}")
                
                # ML Recommendations Section
                if results.get('ml_recommendations'):
                    st.markdown("---")
                    st.markdown("### 🤖 ML Advisor Recommendations")
                    
                    ml_rec = results['ml_recommendations']
                    ml_use_case = ml_rec.get('ml_use_case', {})
                    feature_eng = ml_rec.get('feature_engineering', {})
                    
                    # ML Use Case Detection
                    st.markdown("#### 🎯 Detected ML Use Case")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Primary Use Case", ml_use_case.get('detected_use_case', 'N/A'))
                    col2.metric("Target Variable", ml_use_case.get('target_variable', 'N/A'))
                    col3.metric("ML Readiness Score", f"{ml_use_case.get('suitability_score', '0')}/100")
                    
                    st.info(f"**Reasoning**: {ml_use_case.get('target_reasoning', 'N/A')}")
                    
                    if ml_use_case.get('alternative_use_case', 'N/A') != 'N/A':
                        st.warning(f"**Alternative**: {ml_use_case.get('alternative_use_case')}")
                    
                    # Feature Engineering Plan
                    st.markdown("#### 🔧 Feature Engineering Roadmap")
                    
                    with st.expander("📋 Column-by-Column Transformation Plan", expanded=True):
                        st.markdown(feature_eng.get('feature_plan', 'No plan generated'))
                    
                    # Training Strategy
                    st.markdown("#### 🚀 Training Recommendations")
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown("**Model & Validation**")
                        st.write(feature_eng.get('training_recommendations', 'N/A'))
                    
                    with col_right:
                        st.markdown("**MLflow Setup**")
                        st.write(feature_eng.get('mlflow_setup', 'N/A'))
                    
                    # Categorical columns
                    if profile.get('categorical_analysis') and len(profile['categorical_analysis']) > 0:
                        st.markdown("#### 🏷️ Categorical Columns")
                        for col_data in profile['categorical_analysis']:
                            with st.expander(f"**{col_data['column_name']}** - {col_data['cardinality']} unique values"):
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    st.markdown("**📋 Distribution**")
                                    st.write(f"- **Unique Values**: {col_data['cardinality']}")
                                    st.write(f"- **Most Common**: {col_data['top_value']} ({col_data['top_frequency']}x)")
                                    st.write("- **Top 5 Values**:")
                                    for val, count in col_data['top_5']:
                                        st.write(f"  - {val}: {count}")
                                
                                with col_right:
                                    st.markdown("**💡 AI Insights**")
                                    st.info(col_data['insight'])
                                    st.warning(f"🔍 **Suggestion**: {col_data['actionable_suggestion']}")

                # Deployment Strategy Section
                if results.get('deployment_strategy'):
                    st.markdown("---")
                    st.markdown("### 🚀 Deployment & MLOps Strategy")
                    
                    deployment = results['deployment_strategy']
                    
                    # Technical Infrastructure
                    st.markdown("#### 🏗️ Technical Infrastructure")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("⚙️ Databricks Setup", expanded=False):
                            st.markdown(deployment.get('databricks_setup', 'N/A'))
                        
                        with st.expander("🌐 Serving Strategy", expanded=False):
                            st.markdown(deployment.get('serving_strategy', 'N/A'))
                    
                    with col2:
                        with st.expander("📊 Monitoring Plan", expanded=False):
                            st.markdown(deployment.get('monitoring_plan', 'N/A'))
                        
                        with st.expander("💾 Data Strategy", expanded=False):
                            st.markdown(deployment.get('data_strategy', 'N/A'))
                    
                    # Team & Timeline
                    st.markdown("#### 👥 Team & Timeline")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("👨‍💼 Team Requirements", expanded=True):
                            st.markdown(deployment.get('team_requirements', 'N/A'))
                        
                        with st.expander("⚠️ Risk Mitigation", expanded=False):
                            st.markdown(deployment.get('risk_mitigation', 'N/A'))
                    
                    with col2:
                        with st.expander("📅 Implementation Roadmap", expanded=True):
                            st.markdown(deployment.get('implementation_roadmap', 'N/A'))
                    
                    # Governance & Business
                    st.markdown("#### 📋 Governance & Business Impact")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("💰 Cost Estimation", expanded=False):
                            st.markdown(deployment.get('cost_estimation', 'N/A'))
                        
                        with st.expander("🎯 Success Metrics", expanded=False):
                            st.markdown(deployment.get('success_metrics', 'N/A'))
                    
                    with col2:
                        with st.expander("🔒 Governance Framework", expanded=False):
                            st.markdown(deployment.get('governance_framework', 'N/A'))
                        
                        with st.expander("💼 Business Impact", expanded=True):
                            st.markdown(deployment.get('business_impact', 'N/A'))
                    
                    # Operations & Quality
                    st.markdown("#### 🛠️ Operations & Quality")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with st.expander("🧪 Testing Framework", expanded=False):
                            st.markdown(deployment.get('testing_framework', 'N/A'))
                    
                    with col2:
                        with st.expander("🚨 Operational Playbook", expanded=False):
                            st.markdown(deployment.get('operational_playbook', 'N/A'))
                    
                    with col3:
                        with st.expander("📚 Enablement Plan", expanded=False):
                            st.markdown(deployment.get('enablement_plan', 'N/A'))
                    
                    # Future Vision
                    st.markdown("#### 🔮 Future Enhancements")
                    with st.expander("💡 Roadmap & Innovations", expanded=False):
                        st.markdown(deployment.get('future_enhancements', 'N/A'))
                    # Add after Deployment Strategy section

                    # Business Communication Section
                    if results.get('business_communication'):
                        st.markdown("---")
                        st.markdown("### 📊 Executive Communication")
                        
                        biz_comm = results['business_communication']
                        
                        # Executive Summary - Most prominent
                        st.markdown("#### 📋 Executive Summary")
                        with st.expander("🎯 One-Page Project Overview", expanded=True):
                            st.markdown(biz_comm.get('executive_summary', 'N/A'))
                        
                        
                        
                        # Risk Matrix & Budget side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ⚠️ Risk Assessment")
                            with st.expander("📊 Risk Matrix", expanded=False):
                                st.markdown(biz_comm.get('risk_matrix', 'N/A'))
                        
                        with col2:
                            st.markdown("#### 💰 Investment Case")
                            with st.expander("💵 Budget & ROI", expanded=False):
                                st.markdown(biz_comm.get('budget_justification', 'N/A'))
                        
                        # Stakeholder Talking Points
                        st.markdown("#### 💬 Stakeholder Communication")
                        with st.expander("🗣️ Key Messages by Audience", expanded=False):
                            st.markdown(biz_comm.get('stakeholder_talking_points', 'N/A'))

                    # After all existing sections, add PRD generation section at the very end

                    # PRD Generation Section
                    if results.get('business_communication') and results['status'] == 'completed':
                        st.markdown("---")
                        st.markdown("### 📝 Product Requirements Document (PRD)")
                        
                        st.info("👆 Review all analysis above, then generate the complete PRD document")
                        
                        # Generate PRD Button
                        if st.button("🚀 Generate PRD", type="primary", use_container_width=True):
                            #st.write("DEBUG: Before PRD generation - session_state keys:", list(st.session_state.keys()))
                            #st.write("DEBUG: Analysis Results Available?", 'analysis_results' in st.session_state)
                            with st.spinner("📝 Generating comprehensive PRD..."):
                                prd_result = supervisor.generate_prd(st.session_state.analysis_results)
                                st.session_state.prd_result = prd_result
                        
                        # Display PRD if generated
                        if 'prd_result' in st.session_state and st.session_state.prd_result:
                            prd_data = st.session_state.prd_result
                            
                            if prd_data['status'] == 'success':
                                st.success("✅ PRD Generated Successfully!")
                                
                                # Display PRD
                                st.markdown("#### 📄 Complete PRD Document")
                                with st.expander("📋 View Full PRD", expanded=True):
                                    st.markdown(prd_data['prd_document'])
                                
                                # Download Button
                                st.download_button(
                                    label="⬇️ Download PRD as Markdown",
                                    data=prd_data['prd_document'],
                                    file_name="ML_Product_Requirements_Document.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ PRD generation failed. Please review the error above.")                
                # Show any errors
                if results['errors']:
                    st.error("⚠️ Errors encountered:")
                    for error in results['errors']:
                        st.write(f"- {error}")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        st.markdown("""
        ### 📌 Getting Started
        1. Example : Download the **Superstore** dataset from [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
        2. Upload the `Sample - Superstore.csv`/ ANY file above
        3. Click "Analyze Schema" to see the AI agent in action
        
        ### 🎯 What This Does
        - Uses **pandas** to extract technical schema details
        - Uses **GPT-4** (via DSPy) to interpret business meaning
        """)


if __name__ == "__main__":
    main()