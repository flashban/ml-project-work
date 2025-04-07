import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model(pipeline, X_train, X_test, y_train, y_test):
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    # Generate predictions on test data
    y_pred = pipeline.predict(X_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # Save the trained model
    joblib.dump(pipeline, 'models/co2_emissions_model.pkl')
    return {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}

def load_model(model_path):
    # Load a saved model from file
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_emissions(model, input_data):
    # Predict CO2 emissions using the model
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None