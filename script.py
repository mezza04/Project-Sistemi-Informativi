import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import os
import streamlit as st

def load_data():
    data = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real estate valuation data set.xlsx"))
    #data.sample(3)
    #data.isnull().sum()
    data = data.rename(columns={'Y house price of unit area': 'Price', 
                                'X1 transaction date': 'Date', 
                                'X2 house age': 'House_age', 
                                'X3 distance to the nearest MRT station': 'Station_distance',
                                'X4 number of convenience stores': 'Stores_number',
                                'X5 latitude': 'Latitude',
                                'X6 longitude': 'Longitude'
                                })
    return data

data = load_data()
lat_min = data['Latitude'].min()
lat_max = data['Latitude'].max()
long_min = data['Longitude'].min()
long_max = data['Longitude'].max()


# ADDESTRAMENTO MODELLI 

def train_model(grid_search=False):
    # Split target and predictor
    X = data[['Latitude', 'Longitude']]
    y = data['Price']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Gradient Boosting Regressor
    gradient_boosting_model = GradientBoostingRegressor(random_state=42)
    gradient_boosting_model.fit(X_train, y_train)
    
    return gradient_boosting_model
    
    # Predict on test set
    #y_pred = gradient_boosting_model.predict(X_test)
    
    # Evaluation metrics
    #mae = mean_absolute_error(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    #rmse = np.sqrt(mse)
    #r2 = r2_score(y_test, y_pred)

    #print(f"Mean Absolute Error (MAE): {mae}")
    #print(f"Mean Squared Error (MSE): {mse}")
    #print(f"Root Mean Squared Error (RMSE): {rmse}")
    #print(f"R^2 Score: {r2}")

gradient_boosting_model = train_model()

def train_model_2(grid_search=False):
    # Split target and predictor
    X_2 = data[['House_age', 'Station_distance', 'Stores_number']]
    y_2 = data['Price']
    
    # Splitting the dataset into training and testing sets
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

    # Initialize and train Gradient Boosting Regressor
    gradient_boosting_model_2 = GradientBoostingRegressor(random_state=42)
    gradient_boosting_model_2.fit(X_train_2, y_train_2)
    
    return gradient_boosting_model_2

    # Predict on test set
    #y_pred_2 = gradient_boosting_model_2.predict(X_test_2)
    
    # Evaluation metrics
    #mae_2 = mean_absolute_error(y_test_2, y_pred_2)
    #mse_2 = mean_squared_error(y_test_2, y_pred_2)
    #rmse_2 = np.sqrt(mse_2)
    #r2_2 = r2_score(y_test_2, y_pred_2)
    
    #print(f"Mean Absolute Error (MAE): {mae_2}")
    #print(f"Mean Squared Error (MSE): {mse_2}")
    #print(f"Root Mean Squared Error (RMSE): {rmse_2}")
    #print(f"R^2 Score: {r2_2}")

gradient_boosting_model_2 = train_model_2()


# INTERFACCIA

st.title("Real Estate Prices Estimator")
st.write("##### Specific for Sindian Dist., New Taipei City and Taiwan.")

st.markdown("---")

modello_scelto = st.radio("Select the model used for the prediction (different models require different predictions):",
                          ('Model 1', 'Model 2'))

st.markdown("---")

st.write("Please fill the required slot to determine the house price.")

if modello_scelto == 'Model 1':
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Latitud**  \n*(Valid values: {lat_min:.6f} – {lat_max:.6f})*")
        user_latitude = st.number_input(
            label="",  
            min_value=float(lat_min),
            max_value=float(lat_max),
            format="%.6f",
            key="user_latitude_input"
            )
    with col2:
        st.markdown(f"**Longitude**  \n*(Valid values: {long_min:.6f} – {long_max:.6f})*")
        user_longitude = st.number_input(
            label="", 
            min_value=float(long_min),
            max_value=float(long_max),
            format="%.6f",
            key="user_longitude_input"
            )

    user_input = np.array([user_latitude, user_longitude])

    if st.button("Predict Price"):
        prediction = gradient_boosting_model.predict(user_input.reshape(1, -1))
        st.success(f"Predicted House Price in New Taiwan Dollar/Ping: {prediction}")

elif modello_scelto == 'Model 2':
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Number of years since the house was built**  \n*(Value must be positive)*")
        user_age = st.number_input(
            label="",
            min_value=0.0,
            format="%.1f",
            key="user_age_input"
            )
    with col2:
        st.markdown(f"**Distance from the nearest MRT station**  \n*(Value must be positive)*")
        user_distance = st.number_input(
            label="",
            min_value=0.0,
            format="%.6f",
            key="user_distance_input"
            )
    with col3:
        st.markdown(f"**Number of convenience store near the house**  \n*(Value must be integer)*")
        user_number = st.number_input(
            label="",
            min_value=0,
            step=1,
            key="user_number_input"
            )

    user_input_2 = np.array([user_age, user_distance, user_number])

    if st.button("Predict Price"):
        prediction_2 = gradient_boosting_model_2.predict(user_input_2.reshape(1, -1))
        st.success(f"Predicted House Price in New Taiwan Dollar/Ping:  {prediction_2}")
        
        

