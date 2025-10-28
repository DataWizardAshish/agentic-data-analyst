"""
Profile Agent - Statistical profiling with insights
Hybrid: pandas statistics + DSPy interpretation
"""
import pandas as pd
import numpy as np
import dspy
from signatures.dspy_signatures import StatisticalInsightGenerator


class ProfileAgent:
    """
    Analyzes statistical properties of dataset columns
    - Numeric: mean, median, std, distribution patterns
    - Categorical: cardinality, top values, distribution
    - Uses DSPy to generate business insights
    """
    
    def __init__(self):
        """Initialize DSPy insight generator"""
        self.insight_generator = dspy.ChainOfThought(StatisticalInsightGenerator)
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Profile dataframe columns
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict with numeric_analysis and categorical_analysis
        """
        results = {
            'numeric_analysis': [],
            'categorical_analysis': []
        }
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze numeric columns
        for col in numeric_cols:
            col_analysis = self._analyze_numeric_column(df, col)
            results['numeric_analysis'].append(col_analysis)
        
        # Analyze categorical columns
        for col in categorical_cols:
            col_analysis = self._analyze_categorical_column(df, col)
            results['categorical_analysis'].append(col_analysis)
        
        return results
    
    def _analyze_numeric_column(self, df: pd.DataFrame, col: str) -> dict:
        """
        Analyze numeric column with statistics + LLM insights
        """
        # PROGRAMMATIC PART (70%)
        stats = df[col].describe()
        mean_val = float(stats['mean'])
        median_val = float(df[col].median())
        std_val = float(stats['std'])
        min_val = float(stats['min'])
        max_val = float(stats['max'])
        q25 = float(stats['25%'])
        q75 = float(stats['75%'])
        
        # Calculate skewness
        skewness = float(df[col].skew())
        
        # Detect pattern
        if abs(skewness) < 0.5:
            pattern = "normal distribution"
        elif skewness > 0.5:
            pattern = "right skewed"
        else:
            pattern = "left skewed"
        
        # Build stats dict for LLM
        stats_dict = {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'skewness': skewness,
            'q25': q25,
            'q75': q75
        }
        
        # LLM INTERPRETATION (30%)
        try:
            interpretation = self.insight_generator(
                column_name=col,
                column_type='numeric',
                stats_dict=str(stats_dict)
            )
            
            insight = interpretation.insight
            pattern_detected = interpretation.pattern_detected
            suggestion = interpretation.actionable_suggestion
            
        except Exception as e:
            insight = f"Error generating insight: {str(e)}"
            pattern_detected = pattern
            suggestion = "Review statistics manually"
        
        return {
            'column_name': col,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'q25': q25,
            'q75': q75,
            'skewness': skewness,
            'pattern_detected': pattern_detected,
            'insight': insight,
            'actionable_suggestion': suggestion
        }
    
    def _analyze_categorical_column(self, df: pd.DataFrame, col: str) -> dict:
        """
        Analyze categorical column with value counts + LLM insights
        """
        # PROGRAMMATIC PART (70%)
        value_counts = df[col].value_counts()
        cardinality = len(value_counts)
        total_count = len(df[col].dropna())
        
        top_value = value_counts.index[0] if len(value_counts) > 0 else "N/A"
        top_frequency = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        
        # Get top 5 values
        top_5 = [(str(val), int(count)) for val, count in value_counts.head(5).items()]
        
        # Build stats dict for LLM
        stats_dict = {
            'cardinality': cardinality,
            'total_count': total_count,
            'top_value': str(top_value),
            'top_frequency': top_frequency,
            'top_5_values': str(top_5)
        }
        
        # LLM INTERPRETATION (30%)
        try:
            interpretation = self.insight_generator(
                column_name=col,
                column_type='categorical',
                stats_dict=str(stats_dict)
            )
            
            insight = interpretation.insight
            pattern_detected = interpretation.pattern_detected
            suggestion = interpretation.actionable_suggestion
            
        except Exception as e:
            insight = f"Error generating insight: {str(e)}"
            pattern_detected = "unknown"
            suggestion = "Review distribution manually"
        
        return {
            'column_name': col,
            'cardinality': cardinality,
            'top_value': str(top_value),
            'top_frequency': top_frequency,
            'top_5': top_5,
            'pattern_detected': pattern_detected,
            'insight': insight,
            'actionable_suggestion': suggestion
        }