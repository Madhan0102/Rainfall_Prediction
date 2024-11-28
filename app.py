import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the machine learning model
model_file = 'rainfall_prediction_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Weather Prediction Application")

# Collect user inputs
st.sidebar.header("Input Parameters")
pressure = st.sidebar.number_input("Pressure (hPa)", min_value=0.0, value=1013.25, step=0.1)
dewpoint = st.sidebar.number_input("Dewpoint (°C)", min_value=-100.0, max_value=100.0, value=10.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
cloud = st.sidebar.slider("Cloud Cover (%)", min_value=0, max_value=100, value=50, step=1)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
sunshine = st.sidebar.number_input("Sunshine (hours)", min_value=0.0, value=6.0, step=0.1)
winddirection = st.sidebar.slider("Wind Direction (°)", min_value=0, max_value=360, value=90, step=1)
windspeed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0, step=0.1)

# Button for prediction
if st.button("Predict"):
    # Prepare the input for prediction
    input_data = np.array([[pressure, dewpoint, humidity, cloud, rainfall, sunshine, winddirection, windspeed]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    st.subheader("Prediction Result")
    st.write(f"The predicted value is: {prediction[0]}")

    # Plotting the prediction
    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots()
    ax.bar(["Prediction"], [prediction[0]])
    ax.set_ylabel("Predicted Value")
    st.pyplot(fig)

st.markdown("Developed with ❤️ by [Madhan Kumar N]")
