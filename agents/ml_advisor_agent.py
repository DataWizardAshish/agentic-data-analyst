"""
ML Advisor Agent - Synthesizes analysis into ML recommendations
Provides use case detection, target selection, and feature engineering plan
"""

import dspy

from signatures.dspy_signatures import (FeatureEngineeringPlanner,
                                        MLUseCaseDetector)


class MLAdvisorAgent:
    """
    Analyzes schema, profile, and quality results to generate:
    - ML use case recommendations
    - Target variable suggestions
    - Feature engineering roadmap
    - Training strategy
    """

    def __init__(self):
        self.detector = dspy.ChainOfThought(MLUseCaseDetector)
        self.planner = dspy.ChainOfThought(FeatureEngineeringPlanner)

    def analyze(
        self, schema_results: dict, profile_results: dict, quality_results: dict
    ) -> dict:
        """
        Generate ML recommendations from previous agent outputs

        Args:
            schema_results: Output from SchemaAgent
            profile_results: Output from ProfileAgent
            quality_results: Output from QualityAgent

        Returns:
            Dict with ml_use_case and feature_engineering plans
        """
        # Step 1: Convert complex agent outputs to concise summaries
        dataset_overview = self._create_dataset_overview(schema_results)
        key_columns = self._extract_key_columns(schema_results, profile_results)
        quality_summary = self._summarize_quality_issues(quality_results)

        # Step 2: Detect ML use case and target variable
        try:
            ml_detection = self.detector(
                dataset_overview=dataset_overview,
                key_columns=key_columns,
                quality_issues=quality_summary,
            )

            detected_use_case = ml_detection.detected_use_case
            target_variable = ml_detection.target_variable
            target_reasoning = ml_detection.target_reasoning
            suitability_score = ml_detection.suitability_score
            alternative_use_case = ml_detection.alternative_use_case

        except Exception as e:
            print(f"❌ ML Use Case Detection failed: {str(e)}")
            detected_use_case = "Unable to detect"
            target_variable = "Unknown"
            target_reasoning = f"Error: {str(e)}"
            suitability_score = "0"
            alternative_use_case = "N/A"

        # Step 3: Generate feature engineering plan
        try:
            column_summary = self._create_column_summary(
                schema_results, profile_results
            )
            instructions = self._get_use_case_instructions(detected_use_case)

            feature_planning = self.planner(
                column_summary=column_summary,
                target_variable=target_variable,
                ml_use_case=detected_use_case,
                planning_instructions=instructions,
            )

            feature_plan = feature_planning.feature_plan
            training_recommendations = feature_planning.training_recommendations
            mlflow_setup = feature_planning.mlflow_setup

        except Exception as e:
            print(f"❌ Feature Engineering Planning failed: {str(e)}")
            feature_plan = f"Error generating plan: {str(e)}"
            training_recommendations = "Unable to generate recommendations"
            mlflow_setup = "Unable to generate MLflow recommendations"

        # Consolidate results
        return {
            "ml_use_case": {
                "detected_use_case": detected_use_case,
                "target_variable": target_variable,
                "target_reasoning": target_reasoning,
                "suitability_score": suitability_score,
                "alternative_use_case": alternative_use_case,
            },
            "feature_engineering": {
                "feature_plan": feature_plan,
                "training_recommendations": training_recommendations,
                "mlflow_setup": mlflow_setup,
            },
        }

    def _get_use_case_instructions(self, use_case: str) -> str:
        """Generate use case-specific instructions"""
        base = """You are an expert data scientist. Generate a clear, step-by-step ML plan in MARKDOWN FORMAT:

    ## 1. Data Preparation
    - Train/validation/test splits (ratios)
    - Preprocessing steps
    - Feature transformations

    ## 2. Model Training
    - Baseline model (specify)
    - Advanced models (specify algorithms)
    - Training sequence

    ## 3. Evaluation & Validation
    - Primary metrics (specify)
    - Cross-validation strategy
    - Holdout evaluation approach

    ## 4. Hyperparameter Tuning
    - Key parameters to tune
    - Search strategy (grid/random/bayesian)

    ## 5. MLflow Tracking
    - Experiment setup code
    - Parameters and metrics to log
    - Artifact storage plan

    ## 6. Deployment & Monitoring
    - Model serialization format
    - Monitoring metrics
    - Retraining triggers

    Use markdown headers (##), bullet points (-), keep responses concise and actionable."""

        use_case_lower = use_case.lower()
        if "classification" in use_case_lower:
            return (
                base
                + "\n\n**Classification Focus:** Include class imbalance handling, precision/recall tradeoffs, ROC-AUC, confusion matrix analysis."
            )
        elif "regression" in use_case_lower:
            return (
                base
                + "\n\n**Regression Focus:** Emphasize RMSE, MAE, R², residual analysis, outlier detection and handling."
            )
        elif "clustering" in use_case_lower:
            return (
                base
                + "\n\n**Clustering Focus:** Include silhouette score, elbow method, feature scaling requirements, cluster interpretation."
            )
        else:
            return base

    def _create_dataset_overview(self, schema_results: dict) -> str:
        """Create concise dataset overview string"""
        summary = schema_results["summary"]
        return f"Dataset: {summary['total_rows']} rows, {summary['total_columns']} columns, {summary['memory_usage_mb']:.1f}MB"

    def _extract_key_columns(self, schema_results: dict, profile_results: dict) -> str:
        """Extract most relevant columns (limit to top 10 to save tokens)"""
        columns = schema_results["columns"]

        # Prioritize: low null%, high uniqueness, non-text types
        key_cols = []
        for col in columns[:10]:  # Limit to 10 columns
            col_info = f"{col['column_name']} ({col['business_type']}, {col['null_percentage']}% nulls, {col['unique_count']} unique)"
            key_cols.append(col_info)

        return "; ".join(key_cols)

    def _summarize_quality_issues(self, quality_results: dict) -> str:
        """Summarize quality issues concisely"""
        summary = quality_results.get("summary", {})
        total = summary.get("total_issues", 0)
        critical = summary.get("critical", 0)
        warnings = summary.get("warnings", 0)

        if total == 0:
            return "No quality issues detected"

        return f"{total} issues found: {critical} critical, {warnings} warnings"

    def _create_column_summary(
        self, schema_results: dict, profile_results: dict
    ) -> str:
        """Create column summary for feature engineering (top 15 columns max)"""
        columns = schema_results["columns"][:15]  # Limit to prevent token overflow

        summary_lines = []
        for col in columns:
            line = (
                f"- {col['column_name']}: {col['business_type']}, {col['pandas_dtype']}"
            )

            # Add cardinality info
            if col["unique_count"] < 20:
                line += f", {col['unique_count']} categories"
            elif col["unique_count"] == len(columns):
                line += ", unique identifier"

            # Add null info if significant
            if col["null_percentage"] > 5:
                line += f", {col['null_percentage']}% nulls"

            summary_lines.append(line)

        return "\n".join(summary_lines)
