import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Import utility functions
from utils import (
    load_dataset, 
    create_feature_description, 
    validate_input_data, 
    log_prediction
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    """
    Load the pre-trained machine learning model
    
    Returns:
        model: Loaded machine learning model
    """
    try:
        model = joblib.load('../models/co2_emissions_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Page configuration
    st.set_page_config(page_title="CO2 Emissions Predictor", page_icon="🚗")
    
    # Title
    st.title("🌍 Vehicle CO2 Emissions Predictor")
    
    # Load dataset and model
    df = load_dataset('../data/CO2 Emissions_Canada.csv')
    model = load_model()
    
    if df is None or model is None:
        st.error("Failed to load dataset or model. Please check your files.")
        return
    
    # Sidebar feature descriptions
    with st.sidebar.expander("Feature Descriptions"):
        feature_desc = create_feature_description()
        for feature, description in feature_desc.items():
            st.sidebar.markdown(f"**{feature}**: {description}")
    
    # Sidebar for input features
    st.sidebar.header("Input Vehicle Specifications")
    
    # Prepare valid categories for validation
    valid_categories = {
        'Make': sorted(df['Make'].unique()),
        'Model': sorted(df['Model'].unique()),
        'Vehicle Class': sorted(df['Vehicle Class'].unique()),
        'Transmission': sorted(df['Transmission'].unique()),
        'Fuel Type': sorted(df['Fuel Type'].unique())
    }
    
    # Input features
    make = st.sidebar.selectbox("Vehicle Make", valid_categories['Make'])
    
    # Filter models based on selected make
    models_filtered = sorted(df[df['Make'] == make]['Model'].unique())
    model_selected = st.sidebar.selectbox("Vehicle Model", models_filtered)
    
    # Other inputs
    vehicle_class = st.sidebar.selectbox(
        "Vehicle Class", 
        valid_categories['Vehicle Class']
    )
    
    engine_size = st.sidebar.slider(
        "Engine Size (L)", 
        float(df['Engine Size(L)'].min()), 
        float(df['Engine Size(L)'].max()), 
        float(df['Engine Size(L)'].median())
    )
    
    cylinders = st.sidebar.slider(
        "Number of Cylinders", 
        int(df['Cylinders'].min()), 
        int(df['Cylinders'].max()), 
        int(df['Cylinders'].median())
    )
    
    transmission = st.sidebar.selectbox(
        "Transmission", 
        valid_categories['Transmission']
    )
    
    fuel_type = st.sidebar.selectbox(
        "Fuel Type", 
        valid_categories['Fuel Type']
    )
    
    # Prediction
    if st.sidebar.button("Predict CO2 Emissions"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Make': [make],
            'Model': [model_selected],
            'Vehicle Class': [vehicle_class],
            'Engine Size(L)': [engine_size],
            'Cylinders': [cylinders],
            'Transmission': [transmission],
            'Fuel Type': [fuel_type]
        })
        
        # Validate input data
        if validate_input_data(input_data, valid_categories):
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Log the prediction
            log_prediction(input_data, prediction)
            
            # Display prediction
            st.success(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
            
            # Emissions interpretation
            if prediction < 200:
                st.info("Low emissions: Environmentally friendly vehicle")
            elif prediction < 300:
                st.warning("Moderate emissions: Consider more efficient options")
            else:
                st.error("High emissions: Significant environmental impact")
            
            # Visualization
            st.subheader("Emissions Distribution")
            fig = px.box(df, x='Fuel Type', y='CO2 Emissions(g/km)', 
                         title='CO2 Emissions by Fuel Type')
            st.plotly_chart(fig)
        else:
            st.error("Invalid input data. Please check your selections.")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    🚗 CO2 Emissions Predictor
    
    Predict vehicle CO2 emissions based on specifications.
    Helps in understanding environmental impact of vehicles.
    """)

if __name__ == "__main__":
    main()