import pandas as pd
import numpy as np
from engines.recommender_helpers import sector_pca_visualize
from sklearn.decomposition import PCA

def test_enhanced_hover_visualization():
    """
    Test the enhanced PCA visualization with correlation scores in hover tooltips
    """
    print("=== Testing Enhanced PCA Visualization with Correlation in Hover ===")
    
    # Create sample data for testing
    sample_companies = [
        'Microsoft Corporation',
        'Apple Inc.',
        'Google LLC',
        'Amazon.com Inc.',
        'Tesla Inc.',
        'Johnson & Johnson',
        'Pfizer Inc.',
        'JPMorgan Chase',
        'Walmart Inc.',
        'Coca-Cola Company'
    ]
    
    # Create sample sectors
    sectors = [
        'TMT', 'TMT', 'TMT', 'TMT', 'TMT',
        'Healthcare', 'Healthcare', 'Financial', 'Consumer', 'Consumer'
    ]
    
    # Create sample correlation scores
    scores = [0.95, 0.92, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77, 0.75]
    
    # Create sample PCA data (2D coordinates)
    np.random.seed(42)  # For reproducible results
    X_reduced = np.random.randn(len(sample_companies), 2)
    
    print("Sample companies with correlation scores:")
    for i, (company, score) in enumerate(zip(sample_companies, scores)):
        print(f"  {i+1:2d}. {company}: {score:.3f}")
    
    print("\nHover over companies in the visualization to see:")
    print("- Company name (bold)")
    print("- Sector information")
    print("- Correlation/Similarity score")
    
    # Test the enhanced visualization with correlation in hover
    sector_pca_visualize(X_reduced, sectors, sample_companies, scores, 'correlation')
    
    print("\n=== Visualization completed ===")
    print("Features:")
    print("- Color coding: By sector (gray = No Sector)")
    print("- Hover tooltips: Company name, sector, and correlation score")
    print("- Interactive: Zoom, pan, and hover for details")

if __name__ == "__main__":
    test_enhanced_hover_visualization()