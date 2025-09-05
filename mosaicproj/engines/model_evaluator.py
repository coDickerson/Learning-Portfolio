import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """
    Evaluates the accuracy of the recommender engine using train/validation splits.
    """
    
    def __init__(self, company_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            company_df: DataFrame with columns ['source_company', 'target_company', 'request_date']
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.company_df = company_df.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        
    def split_data_by_company(self, company_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data for a specific company into training and validation sets.
        
        Args:
            company_id: ID of the company to evaluate
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        # Get all interactions for this company
        company_data = self.company_df[self.company_df['source_company'] == company_id].copy()
        
        if len(company_data) < 2:
            raise ValueError(f"Company {company_id} has insufficient data for splitting (need at least 2 interactions)")
        
        # Sort by date to ensure temporal consistency
        company_data = company_data.sort_values('request_date')
        
        # Split the interactions
        train_data, val_data = train_test_split(
            company_data, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=None  # No stratification since we want random split
        )
        
        return train_data, val_data
    
    def evaluate_recommendations(self, 
                               company_id: int, 
                               method: str = 'multivector',
                               top_n: int = 10,
                               threshold: float = 0.4) -> Dict[str, Any]:
        """
        Evaluate recommendation accuracy for a specific company.
        
        Args:
            company_id: ID of the company to evaluate
            method: Recommendation method ('pairwise' or 'multivector')
            top_n: Number of top recommendations to consider
            threshold: Correlation threshold for pairwise method
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Split data
            train_data, val_data = self.split_data_by_company(company_id)
            
            # Get validation set companies
            val_companies = set(val_data['target_company'].unique())
            
            # Create temporary full dataset with only training data
            temp_df = self.company_df[self.company_df['source_company'] != company_id].copy()
            temp_df = pd.concat([temp_df, train_data])
            
            # Generate recommendations using training data
            from engines.recommender_engine import recommend
            recommendations = recommend(
                temp_df, 
                company_id, 
                method=method, 
                threshold=threshold, 
                top_n=top_n
            )
            
            if recommendations.empty:
                return {
                    'company_id': company_id,
                    'method': method,
                    'success': False,
                    'error': 'No recommendations generated',
                    'metrics': {}
                }
            
            # Get top N recommended companies
            recommended_companies = set(recommendations.head(top_n).index.tolist())
            
            # Calculate metrics
            hits = val_companies.intersection(recommended_companies)
            precision = len(hits) / len(recommended_companies) if recommended_companies else 0
            recall = len(hits) / len(val_companies) if val_companies else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate additional metrics
            hit_rate = len(hits) / len(val_companies) if val_companies else 0
            coverage = len(recommended_companies) / len(self.company_df['target_company'].unique())
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'hit_rate': hit_rate,
                'coverage': coverage,
                'hits': len(hits),
                'total_val_companies': len(val_companies),
                'total_recommendations': len(recommended_companies),
                'val_companies': list(val_companies),
                'recommended_companies': list(recommended_companies),
                'hits_companies': list(hits)
            }
            
            return {
                'company_id': company_id,
                'method': method,
                'success': True,
                'metrics': metrics,
                'train_size': len(train_data),
                'val_size': len(val_data)
            }
            
        except Exception as e:
            return {
                'company_id': company_id,
                'method': method,
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def evaluate_multiple_companies(self, 
                                  company_ids: List[int], 
                                  method: str = 'multivector',
                                  top_n: int = 10,
                                  threshold: float = 0.4) -> pd.DataFrame:
        """
        Evaluate multiple companies and return aggregated results.
        
        Args:
            company_ids: List of company IDs to evaluate
            method: Recommendation method
            top_n: Number of top recommendations
            threshold: Correlation threshold
            
        Returns:
            DataFrame with evaluation results for all companies
        """
        results = []
        
        for company_id in company_ids:
            result = self.evaluate_recommendations(company_id, method, top_n, threshold)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Extract metrics for successful evaluations
        successful_results = results_df[results_df['success'] == True]
        
        if not successful_results.empty:
            # Calculate aggregate metrics
            metrics_df = pd.DataFrame(successful_results['metrics'].tolist())
            aggregate_metrics = {
                'avg_precision': metrics_df['precision'].mean(),
                'avg_recall': metrics_df['recall'].mean(),
                'avg_f1_score': metrics_df['f1_score'].mean(),
                'avg_hit_rate': metrics_df['hit_rate'].mean(),
                'avg_coverage': metrics_df['coverage'].mean(),
                'total_hits': metrics_df['hits'].sum(),
                'total_val_companies': metrics_df['total_val_companies'].sum(),
                'successful_evaluations': len(successful_results),
                'total_evaluations': len(results)
            }
            
            print(f"\n=== AGGREGATE EVALUATION RESULTS ===")
            print(f"Method: {method}")
            print(f"Top N: {top_n}")
            print(f"Successful evaluations: {aggregate_metrics['successful_evaluations']}/{aggregate_metrics['total_evaluations']}")
            print(f"Average Precision: {aggregate_metrics['avg_precision']:.3f}")
            print(f"Average Recall: {aggregate_metrics['avg_recall']:.3f}")
            print(f"Average F1-Score: {aggregate_metrics['avg_f1_score']:.3f}")
            print(f"Average Hit Rate: {aggregate_metrics['avg_hit_rate']:.3f}")
            print(f"Total Hits: {aggregate_metrics['total_hits']}")
            print(f"Total Validation Companies: {aggregate_metrics['total_val_companies']}")
        
        return results_df
    
    def cross_validate_recommendations(self, 
                                     company_ids: List[int], 
                                     method: str = 'multivector',
                                     top_n: int = 10,
                                     threshold: float = 0.4,
                                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on recommendation accuracy.
        
        Args:
            company_ids: List of company IDs to evaluate
            method: Recommendation method
            top_n: Number of top recommendations
            threshold: Correlation threshold
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        cv_results = []
        
        for fold in range(cv_folds):
            # Set different random state for each fold
            self.random_state = 42 + fold
            
            # Evaluate all companies for this fold
            fold_results = self.evaluate_multiple_companies(
                company_ids, method, top_n, threshold
            )
            
            # Calculate fold metrics
            successful_fold = fold_results[fold_results['success'] == True]
            if not successful_fold.empty:
                metrics_df = pd.DataFrame(successful_fold['metrics'].tolist())
                fold_metrics = {
                    'fold': fold + 1,
                    'precision': metrics_df['precision'].mean(),
                    'recall': metrics_df['recall'].mean(),
                    'f1_score': metrics_df['f1_score'].mean(),
                    'hit_rate': metrics_df['hit_rate'].mean()
                }
                cv_results.append(fold_metrics)
        
        # Calculate cross-validation statistics
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            cv_stats = {
                'mean_precision': cv_df['precision'].mean(),
                'std_precision': cv_df['precision'].std(),
                'mean_recall': cv_df['recall'].mean(),
                'std_recall': cv_df['recall'].std(),
                'mean_f1': cv_df['f1_score'].mean(),
                'std_f1': cv_df['f1_score'].std(),
                'mean_hit_rate': cv_df['hit_rate'].mean(),
                'std_hit_rate': cv_df['hit_rate'].std(),
                'cv_folds': len(cv_results)
            }
            
            print(f"\n=== CROSS-VALIDATION RESULTS ===")
            print(f"Method: {method}")
            print(f"CV Folds: {cv_stats['cv_folds']}")
            print(f"Precision: {cv_stats['mean_precision']:.3f} ± {cv_stats['std_precision']:.3f}")
            print(f"Recall: {cv_stats['mean_recall']:.3f} ± {cv_stats['std_recall']:.3f}")
            print(f"F1-Score: {cv_stats['mean_f1']:.3f} ± {cv_stats['std_f1']:.3f}")
            print(f"Hit Rate: {cv_stats['mean_hit_rate']:.3f} ± {cv_stats['std_hit_rate']:.3f}")
            
            return cv_stats
        else:
            print("No successful cross-validation results")
            return {}
    
    def compare_methods(self, 
                       company_ids: List[int], 
                       top_n: int = 10,
                       threshold: float = 0.4) -> pd.DataFrame:
        """
        Compare the performance of different recommendation methods.
        
        Args:
            company_ids: List of company IDs to evaluate
            top_n: Number of top recommendations
            threshold: Correlation threshold
            
        Returns:
            DataFrame comparing method performance
        """
        methods = ['pairwise', 'multivector']
        comparison_results = []
        
        for method in methods:
            print(f"\nEvaluating {method} method...")
            results = self.evaluate_multiple_companies(company_ids, method, top_n, threshold)
            
            successful_results = results[results['success'] == True]
            if not successful_results.empty:
                metrics_df = pd.DataFrame(successful_results['metrics'].tolist())
                method_summary = {
                    'method': method,
                    'avg_precision': metrics_df['precision'].mean(),
                    'avg_recall': metrics_df['recall'].mean(),
                    'avg_f1_score': metrics_df['f1_score'].mean(),
                    'avg_hit_rate': metrics_df['hit_rate'].mean(),
                    'successful_evaluations': len(successful_results),
                    'total_evaluations': len(results)
                }
                comparison_results.append(method_summary)
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            print(f"\n=== METHOD COMPARISON ===")
            print(comparison_df.to_string(index=False))
            return comparison_df
        else:
            print("No successful method comparisons")
            return pd.DataFrame()

