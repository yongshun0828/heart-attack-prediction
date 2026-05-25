import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Import KNN Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('heartDataSet.csv')
    return df

# Function to make predictions
def predict_heart_attack(data, model):
    prediction = model.predict(data)
    return prediction

# Main App
def main():
    st.title('Heart Attack Prediction App')
    
    # Sidebar for User Inputs
    st.sidebar.header("Input Parameters")
    def user_input_features():
        age = st.sidebar.slider('Age', 20, 80, 50)
        sex = st.sidebar.selectbox('Sex', (0, 1))
        cp = st.sidebar.slider('Chest Pain Type', 0, 3, 1)
        trtbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
        chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 400, 150)
        fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
        restecg = st.sidebar.slider('Resting ECG Results', 0, 2, 1)
        thalachh = st.sidebar.slider('Max Heart Rate Achieved', 60, 200, 120)
        exng = st.sidebar.selectbox('Exercise Induced Angina', (0, 1))
        oldpeak = st.sidebar.slider('Previous Peak (Oldpeak)', 0.0, 6.0, 1.0)
        slp = st.sidebar.slider('Slope of Peak Exercise Segment', 0, 2, 1)
        caa = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 1)
        thall = st.sidebar.slider('Thalassemia Rate (0-3)', 0, 3, 1)

        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall
        }
        return pd.DataFrame(data, index=[0])

    # Load dataset and preprocess
    df = load_data()
    X = df.drop(columns='output')
    y = df['output']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model (K-Nearest Neighbors)
    model = KNeighborsClassifier(n_neighbors=5)  # Example: using 5 neighbors
    model.fit(X_train, y_train)
    
    # Get user input and predict
    input_df = user_input_features()
    scaled_input = scaler.transform(input_df)
    prediction = predict_heart_attack(scaled_input, model)
    
    # Output the prediction
    st.subheader('Prediction:')
    st.write('Heart Attack Risk' if prediction[0] == 1 else 'No Heart Attack Risk')
    
    # Display model performance metrics
    st.subheader('Model Performance:')
    y_pred = model.predict(X_test)
    st.write('Accuracy:', accuracy_score(y_test, y_pred))
    st.write('Precision:', precision_score(y_test, y_pred))
    st.write('Recall:', recall_score(y_test, y_pred))
    st.write('F1 Score:', f1_score(y_test, y_pred))

if __name__ == '__main__':
    main()
