import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10):
    # Extract feature importances from the model
    importances = model.named_steps['regressor'].feature_importances_
    
    # Get feature names after preprocessing (e.g., after one-hot encoding)
    preprocessor = model.named_steps['preprocessor']
    transformed_feature_names = preprocessor.get_feature_names_out()
    
    # Aggregate importances back to original features
    original_features = ['Engine Size(L)', 'Cylinders', 'Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
    aggregated_importances = {feat: 0 for feat in original_features}
    
    for transformed_name, importance in zip(transformed_feature_names, importances):
        # Match transformed feature names to original features
        for orig_feat in original_features:
            if transformed_name.startswith(f'cat__{orig_feat}') or transformed_name.startswith(f'num__{orig_feat}'):
                aggregated_importances[orig_feat] += importance
                break
    
    # Convert to lists for plotting
    feature_names = list(aggregated_importances.keys())
    importances = list(aggregated_importances.values())
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    # Limit to top N features
    sorted_features = sorted_features[:top_n]
    sorted_importances = sorted_importances[:top_n]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Top 10)")
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.tight_layout()
    
    return plt

def plot_actual_vs_predicted(y_test, y_pred):
    # Scatter plot of actual vs predicted values
    fig = go.Figure(data=go.Scatter(
        x=y_test, 
        y=y_pred, 
        mode='markers',
        marker=dict(color='blue', size=10, opacity=0.7)
    ))
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()], 
        y=[y_test.min(), y_test.max()], 
        mode='lines',
        line=dict(color='red', dash='dash')
    ))
    # Customize plot layout
    fig.update_layout(
        title='Actual vs Predicted CO2 Emissions',
        xaxis_title='Actual Emissions (g/km)',
        yaxis_title='Predicted Emissions (g/km)'
    )
    return fig