from src.data_preprocessing import load_data, preprocess_data, prepare_pipeline
from src.model_training import train_model
from src.visualization import plot_feature_importance, plot_actual_vs_predicted
import matplotlib.pyplot as plt

def main():
    # Load data
    data_path = 'data/CO2 Emissions_Canada.csv'
    df = load_data(data_path)
    
    if df is not None:
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        
        # Prepare pipeline
        pipeline = prepare_pipeline(preprocessor)
        
        # Train model
        metrics = train_model(pipeline, X_train, X_test, y_train, y_test)
        
        # Print metrics
        print("Model Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        # Feature importance visualization
        feature_names = (
            preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
            preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
        )
        plot_feature_importance(pipeline, feature_names).savefig('static/images/feature_importance.png')
        
        # Actual vs Predicted plot
        predictions = pipeline.predict(X_test)
        plot_fig = plot_actual_vs_predicted(y_test, predictions)
        plot_fig.write_image('static/images/actual_vs_predicted.png')

if __name__ == "__main__":
    main()