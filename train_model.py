import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Function to generate sample data
def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(loc=38.0, scale=1.0, size=num_samples),
        'breathing_rate': np.random.normal(loc=30.0, scale=5.0, size=num_samples),
        'disease': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])  # 80% healthy, 20% diseased
    }
    df = pd.DataFrame(data)
    return df

# Main function to train the model
def train_model():
    # Generate and preprocess sample data
    df = generate_sample_data()

    # Split data into features and labels
    X = df[['temperature', 'breathing_rate']]
    y = df['disease']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make pdddredictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Save the model and scaler to disk
    with open('animal_disease_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

if __name__ == "__main__":
    train_model()
