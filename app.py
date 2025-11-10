import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Sleep Health Predictor",
    page_icon="😴",
    layout="centered"
)

# Load models
MODEL_PATH = Path(__file__).parent / "models"
model = joblib.load(MODEL_PATH / "model.pkl")
scaler = joblib.load(MODEL_PATH / "scaler.pkl")

# Page title
st.title("Sleep Health Pattern Predictor")
st.write("Enter your details below to find out your sleep health pattern cluster")

# Input fields
col1, col2 = st.columns(2)

with col1:
    social_media = st.number_input(
        "Daily Social Media Usage (minutes)",
        min_value=0,
        max_value=1440,
        value=120
    )

    gaming_hours = st.number_input(
        "Weekly Gaming Hours",
        min_value=0,
        max_value=168,
        value=10
    )

with col2:
    personality = st.slider(
        "Introversion-Extraversion Scale",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Introverted, 5 = Very Extroverted"
    )

# Prediction button
if st.button("Predict Pattern"):
    # Prepare input data
    input_data = np.array([[social_media, gaming_hours, personality]])
    
    # Get cluster and distance
    cluster = model.predict(input_data)[0]
    distances = model.transform(input_data)
    distance_to_center = distances[0][cluster]
    
    # Scale distance
    normalized_distance = scaler.transform([[distance_to_center]])[0][0]
    
    # Display results
    st.subheader("Results")
    
    # Define cluster descriptions
    cluster_descriptions = {
        0: "Balanced Digital User",
        1: "High Screen Time User",
        2: "Minimal Digital User"
    }
    
    st.write(f"**Your Sleep Pattern Group:** {cluster_descriptions[cluster]}")
    
    # Display confidence based on distance
    confidence = (1 - normalized_distance) * 100
    st.write(f"**Pattern Match Confidence:** {confidence:.1f}%")
    
    # Additional insights based on cluster
    st.subheader("Insights")
    if cluster == 0:
        st.write("- You maintain a moderate balance in digital media consumption")
        st.write("- Your habits suggest a healthy relationship with technology")
        st.write("- Consider maintaining this balance for optimal sleep health")
    elif cluster == 1:
        st.write("- Your screen time is higher than average")
        st.write("- Consider reducing evening screen time for better sleep")
        st.write("- Try implementing digital wellness practices")
    else:
        st.write("- You have minimal digital media consumption")
        st.write("- Your habits suggest less exposure to blue light")
        st.write("- This pattern is generally beneficial for sleep health")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Sleep Health Predictor v1.0")