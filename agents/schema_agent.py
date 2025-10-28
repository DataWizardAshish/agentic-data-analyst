"""
Schema Agent - Hybrid approach (Pandas + DSPy)
Analyzes dataframe schema and infers business meaning
"""
import pandas as pd
import dspy
from signatures.dspy_signatures import SchemaInterpreter
from config import MAX_SAMPLE_VALUES


class SchemaAgent:
    """
    Analyzes CSV schema using hybrid approach:
    - Programmatic: Extract pandas dtypes, stats, samples
    - LLM (DSPy): Interpret business meaning and provide recommendations
    """
    
    def __init__(self):
        """Initialize DSPy module for schema interpretation"""
        self.interpreter = dspy.ChainOfThought(SchemaInterpreter)
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze dataframe schema
        
        Args:
            df: pandas DataFrame to analyze
            
        Returns:
            dict with 'columns' list and 'summary' stats
        """
        results = {
            'columns': [],
            'summary': {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        for col in df.columns:
            column_analysis = self._analyze_column(df, col)
            results['columns'].append(column_analysis)
        
        return results
    
    def _analyze_column(self, df: pd.DataFrame, col: str) -> dict:
        """
        Analyze individual column using hybrid approach
        
        Args:
            df: DataFrame
            col: Column name
            
        Returns:
            dict with programmatic stats + LLM interpretation
        """
        # PROGRAMMATIC PART (70% - Rule-based)
        pandas_dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        total_count = len(df)
        
        # Get sample non-null values
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(MAX_SAMPLE_VALUES).tolist()
            # Convert to string representation
            sample_values_str = str(sample_values)
        else:
            sample_values_str = "All null values"
        
        # LLM INTERPRETATION (30% - DSPy reasoning)
        try:
            interpretation = self.interpreter(
                column_name=col,
                pandas_dtype=pandas_dtype,
                null_count=str(null_count),
                unique_count=str(unique_count),
                total_count=str(total_count),
                sample_values=sample_values_str
            )
            
            business_type = interpretation.business_type
            confidence = interpretation.confidence
            reasoning = interpretation.reasoning
            recommendation = interpretation.recommendation
            
        except Exception as e:
            # Fallback if LLM fails
            business_type = "Unknown"
            confidence = "low"
            reasoning = f"Error in LLM interpretation: {str(e)}"
            recommendation = "Review manually"
        
        # Combine programmatic + LLM results
        return {
            'column_name': col,
            'pandas_dtype': pandas_dtype,
            'null_count': null_count,
            'null_percentage': round((null_count / total_count) * 100, 2),
            'unique_count': unique_count,
            'sample_values': sample_values_str,
            'business_type': business_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'recommendation': recommendation
        }