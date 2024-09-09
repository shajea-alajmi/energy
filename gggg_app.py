import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Load the model
model_path = r'C:/Users/shajea/Desktop/solar_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load and clean the data
df = pd.read_csv(r'C:/Users/shajea/Downloads/Solar_Prediction.csv/Solar_Prediction.csv')
df = df.dropna()
df = df[df['Radiation'] >= 0]

# Convert time columns to numeric if necessary
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    except ValueError:
        return np.nan

df['TimeSunRise'] = df['TimeSunRise'].apply(time_to_seconds)
df['TimeSunSet'] = df['TimeSunSet'].apply(time_to_seconds)

# Convert image to base64 for background
def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = load_image_as_base64(r'C:/Users/shajea/Downloads/istockphoto-627281636-612x612.jpg')

# Streamlit app configuration
st.set_page_config(page_title="Solar Radiation Prediction", layout="wide")

# Add custom CSS for background image
st.markdown(f"""
    <style>
    .main {{
        background-image: url(data:image/jpeg;base64,{image_base64});
        background-size: cover;
        background-position: center;
    }}
    .block-container {{
        padding: 2rem;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a section:", ["Home", "Data Exploration", "Model Performance", "Prediction", "About Project"])

if option == "Home":
    st.title("🌞 Solar Radiation Prediction System 🌞")
    st.write("Welcome to the Solar Radiation Prediction System!")
    st.write("""
        **How it works:**
        - The system uses machine learning to predict solar radiation based on input parameters.
        - Explore the data, check model performance, and get predictions using the navigation on the left.
    """)

elif option == "Data Exploration":
    st.title('Data Overview')
    if st.checkbox('Show Raw Data'):
        st.write(df.head())

    # Distribution of solar radiation
    st.subheader('Distribution of Solar Radiation')
    st.write("This histogram shows the distribution of solar radiation values in the dataset. Higher values indicate more intense solar radiation.")
    fig_dist = px.histogram(df, x='Radiation', nbins=50, title='Distribution of Solar Radiation')
    st.plotly_chart(fig_dist)

    # Pairplot of variables
    st.subheader('Pairplot of Variables')
    st.write("This scatter matrix illustrates the relationships between temperature, humidity, and solar radiation. It helps visualize how these variables correlate with each other.")
    fig_pairplot = px.scatter_matrix(df, dimensions=['Temperature', 'Humidity', 'Radiation'],
                                    title='Pairplot of Variables')
    st.plotly_chart(fig_pairplot)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    st.write("This heatmap shows the correlation between temperature, humidity, and solar radiation. Positive values indicate a direct relationship, while negative values indicate an inverse relationship.")
    correlation_matrix = df[['Temperature', 'Humidity', 'Radiation']].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1
    ))
    fig_heatmap.update_layout(title='Correlation Heatmap', xaxis_title='Variables', yaxis_title='Variables')
    st.plotly_chart(fig_heatmap)

    # Time vs Radiation
    st.subheader('Time vs Solar Radiation')
    st.write("This scatter plot shows how solar radiation changes throughout the day based on sunrise and sunset times.")
    df['TimeOfDay'] = df['TimeSunRise'] + (df['TimeSunSet'] - df['TimeSunRise']) / 2  # Midpoint time of day
    fig_time_radiation = px.scatter(df, x='TimeOfDay', y='Radiation', title='Time vs Solar Radiation')
    st.plotly_chart(fig_time_radiation)

    # Pressure vs Wind Speed
    st.subheader('Pressure vs Wind Speed')
    st.write("This scatter plot shows the relationship between atmospheric pressure and wind speed.")
    fig_pressure_speed = px.scatter(df, x='Pressure', y='Speed', title='Pressure vs Wind Speed')
    st.plotly_chart(fig_pressure_speed)

    # Temperature vs Humidity
    st.subheader('Temperature vs Humidity')
    st.write("This scatter plot shows the relationship between temperature and humidity.")
    fig_temp_humidity = px.scatter(df, x='Temperature', y='Humidity', title='Temperature vs Humidity')
    st.plotly_chart(fig_temp_humidity)

elif option == "Model Performance":
    st.title('Model Performance Metrics')
    X = df[['Temperature', 'Humidity', 'Pressure', 'Speed', 'TimeSunRise', 'TimeSunSet']]
    y = df['Radiation']
    y_pred = model.predict(X)
    
    r2 = model.named_steps['regressor'].score(X, y)
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    st.write(f'**R^2 Score:** {r2:.2f}')
    st.write(f'**Mean Absolute Error:** {mae:.2f}')
    st.write(f'**Mean Squared Error:** {mse:.2f}')
    st.write(f'**Root Mean Squared Error:** {rmse:.2f}')
    
    # Plot feature importances
    importances = model.named_steps['regressor'].feature_importances_
    features = X.columns
    st.subheader('Feature Importances')
    st.write("This bar chart shows the importance of each feature in the model. Higher values indicate features that have a greater impact on predicting solar radiation.")
    fig_importances = px.bar(x=importances, y=features, title='Feature Importances')
    st.plotly_chart(fig_importances)

elif option == "Prediction":
    st.title('Solar Radiation Prediction')
    st.sidebar.header('Input Data for Prediction')
    temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
    humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
    pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
    speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
    
    # Input for sunrise and sunset times
    sunrise_time = st.sidebar.time_input("Time of Sunrise", value=datetime.now().time())
    sunset_time = st.sidebar.time_input("Time of Sunset", value=datetime.now().time())
    
    # Convert time to seconds since midnight
    sunrise_seconds = sunrise_time.hour * 3600 + sunrise_time.minute * 60 + sunrise_time.second
    sunset_seconds = sunset_time.hour * 3600 + sunset_time.minute * 60 + sunset_time.second
    
    # Additional input for solar panel details
    st.sidebar.subheader('Solar Panel Specifications')
    panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 10.0, 25.0, 18.0)
    panel_area = st.sidebar.number_input("Panel Area (m²)", value=1.0)
    
    if st.sidebar.button("Predict"):
        input_data = np.array([[temperature, humidity, pressure, speed, sunrise_seconds, sunset_seconds]])
        prediction = model.predict(input_data)[0]
        st.write(f"**Predicted Solar Radiation:** {prediction:.2f} W/m²")
        
        # Calculate expected energy production
        panel_efficiency_decimal = panel_efficiency / 100
        expected_energy = prediction * panel_area * panel_efficiency_decimal
        
        st.subheader('Prediction Details')
        st.write(f"**Temperature:** {temperature}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Pressure:** {pressure} hPa")
        st.write(f"**Wind Speed:** {speed} km/h")
        st.write(f"**Time of Sunrise:** {sunrise_time.strftime('%H:%M:%S')}")
        st.write(f"**Time of Sunset:** {sunset_time.strftime('%H:%M:%S')}")
        st.write(f"**Predicted Radiation:** {prediction:.2f} W/m²")
        st.write(f"**Panel Efficiency:** {panel_efficiency:.2f}%")
        st.write(f"**Panel Area:** {panel_area:.2f} m²")
        st.write(f"**Expected Energy Production:** {expected_energy:.2f} W")

elif option == "About Project":
    st.title('About This Project')
    st.write("""
        **Project Overview:**
        This project aims to predict solar radiation based on various meteorological parameters such as temperature, humidity, pressure, and wind speed. The prediction model is built using machine learning techniques to provide accurate solar radiation estimates, which are crucial for optimizing solar panel performance.

        **Objective:**
        - Predict solar radiation to help in energy management and solar panel efficiency.
        - Provide insights into how different factors influence solar radiation.

        **Data Used:**
        - The dataset includes historical meteorological data and corresponding solar radiation measurements.
        - Key features include temperature, humidity, pressure, wind speed, and sunrise/sunset times.

        **Model:**
        - A machine learning model (e.g., Random Forest Regressor) is trained on the dataset to predict solar radiation based on input features.
        - Model performance is evaluated using metrics such as R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

        **How to Use:**
        - Navigate through the sections to explore data, view model performance, and make predictions.
        - Use the "Prediction" section to input current meteorological conditions and get an estimate of solar radiation.

        **Future Work:**
        - Incorporate additional features such as cloud cover or historical solar radiation data.
        - Improve model accuracy and explore other machine learning algorithms.

        **Contact:**
        For further information or inquiries, please contact the project team at [alajmishajea@gmail.com](madartech0@gmail.com).
    """)
