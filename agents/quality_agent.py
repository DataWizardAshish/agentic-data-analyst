"""
Quality Agent - Data quality checks with recommendations
Hybrid: Rule-based detection + DSPy recommendations
"""
import pandas as pd
import numpy as np
import dspy
from signatures.dspy_signatures import QualityRecommender


class QualityAgent:
    """
    Detects data quality issues:
    - Missing values
    - Duplicates
    - Outliers
    - Inconsistencies
    Uses DSPy to recommend fixes
    """
    
    def __init__(self):
        self.recommender = dspy.ChainOfThought(QualityRecommender)
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """Run all quality checks"""
        results = {
            'issues_found': [],
            'summary': {
                'total_issues': 0,
                'critical': 0,
                'warnings': 0,
                'info': 0
            }
        }
        
        # Check 1: Missing values
        missing_issues = self._check_missing_values(df)
        results['issues_found'].extend(missing_issues)
        
        # Check 2: Duplicates
        duplicate_issues = self._check_duplicates(df)
        results['issues_found'].extend(duplicate_issues)
        
        # Check 3: Outliers
        outlier_issues = self._check_outliers(df)
        results['issues_found'].extend(outlier_issues)
        
        # Check 4: Categorical inconsistencies
        consistency_issues = self._check_categorical_consistency(df)
        results['issues_found'].extend(consistency_issues)
        
        # Update summary counts
        for issue in results['issues_found']:
            results['summary']['total_issues'] += 1
            severity = issue.get('severity', 'info')
            results['summary'][severity] = results['summary'].get(severity, 0) + 1
        
        return results
    
    def _check_missing_values(self, df: pd.DataFrame) -> list:
        """Detect columns with missing values"""
        issues = []
        null_counts = df.isnull().sum()
        
        for col in df.columns:
            null_count = int(null_counts[col])
            if null_count > 0:
                null_pct = (null_count / len(df)) * 100
                
                # Determine severity
                if null_pct > 50:
                    severity = 'critical'
                elif null_pct > 20:
                    severity = 'warnings'
                else:
                    severity = 'info'
                
                # Get LLM recommendation
                issue_desc = f"{null_count} missing values ({null_pct:.1f}%) in column '{col}'"
                sample_data = str(df[col].dropna().head(3).tolist())
                
                try:
                    rec = self.recommender(
                        issue_type='missing_values',
                        column_name=col,
                        issue_description=issue_desc,
                        sample_data=sample_data
                    )
                    action = rec.recommended_action
                    code = rec.python_code
                    impact = rec.impact_description
                except:
                    action = "Impute with median/mode or drop rows"
                    code = f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                    impact = "Fill missing values"
                
                issues.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': severity,
                    'description': issue_desc,
                    'count': null_count,
                    'percentage': round(null_pct, 2),
                    'recommended_action': action,
                    'code_snippet': code,
                    'impact': impact
                })
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> list:
        """Detect duplicate rows"""
        issues = []
        dup_count = df.duplicated().sum()
        
        if dup_count > 0:
            dup_pct = (dup_count / len(df)) * 100
            severity = 'warnings' if dup_pct > 5 else 'info'
            
            issue_desc = f"{dup_count} duplicate rows ({dup_pct:.1f}%)"
            
            try:
                rec = self.recommender(
                    issue_type='duplicates',
                    column_name='entire_row',
                    issue_description=issue_desc,
                    sample_data=str(df[df.duplicated()].head(2).to_dict())
                )
                action = rec.recommended_action
                code = rec.python_code
                impact = rec.impact_description
            except:
                action = "Remove duplicate rows"
                code = "df.drop_duplicates(inplace=True)"
                impact = f"Remove {dup_count} duplicate rows"
            
            issues.append({
                'type': 'duplicates',
                'column': 'entire_row',
                'severity': severity,
                'description': issue_desc,
                'count': dup_count,
                'percentage': round(dup_pct, 2),
                'recommended_action': action,
                'code_snippet': code,
                'impact': impact
            })
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> list:
        """Detect outliers in numeric columns using IQR method"""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                severity = 'warnings' if outlier_pct > 10 else 'info'
                
                issue_desc = f"{outlier_count} outliers ({outlier_pct:.1f}%) in '{col}' (outside [{lower_bound:.2f}, {upper_bound:.2f}])"
                sample_data = str(outliers[col].head(3).tolist())
                
                try:
                    rec = self.recommender(
                        issue_type='outliers',
                        column_name=col,
                        issue_description=issue_desc,
                        sample_data=sample_data
                    )
                    action = rec.recommended_action
                    code = rec.python_code
                    impact = rec.impact_description
                except:
                    action = "Cap outliers or flag for investigation"
                    code = f"df['{col}'] = df['{col}'].clip(lower={lower_bound:.2f}, upper={upper_bound:.2f})"
                    impact = "Cap extreme values"
                
                issues.append({
                    'type': 'outliers',
                    'column': col,
                    'severity': severity,
                    'description': issue_desc,
                    'count': outlier_count,
                    'percentage': round(outlier_pct, 2),
                    'bounds': [round(lower_bound, 2), round(upper_bound, 2)],
                    'recommended_action': action,
                    'code_snippet': code,
                    'impact': impact
                })
        
        return issues
    
    def _check_categorical_consistency(self, df: pd.DataFrame) -> list:
        """Check for inconsistent categorical values"""
        issues = []
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            
            # Check for very similar values (case sensitivity, whitespace)
            if unique_count > 1 and unique_count < 100:
                values_lower = df[col].dropna().str.lower().str.strip()
                normalized_unique = values_lower.nunique()
                
                if normalized_unique < unique_count:
                    diff = unique_count - normalized_unique
                    issue_desc = f"'{col}' has {diff} inconsistent values (case/whitespace issues)"
                    sample_data = str(value_counts.head(5).to_dict())
                    
                    try:
                        rec = self.recommender(
                            issue_type='inconsistent_categories',
                            column_name=col,
                            issue_description=issue_desc,
                            sample_data=sample_data
                        )
                        action = rec.recommended_action
                        code = rec.python_code
                        impact = rec.impact_description
                    except:
                        action = "Standardize categorical values"
                        code = f"df['{col}'] = df['{col}'].str.lower().str.strip()"
                        impact = f"Reduce {diff} redundant categories"
                    
                    issues.append({
                        'type': 'inconsistent_categories',
                        'column': col,
                        'severity': 'info',
                        'description': issue_desc,
                        'count': diff,
                        'recommended_action': action,
                        'code_snippet': code,
                        'impact': impact
                    })
        
        return issues