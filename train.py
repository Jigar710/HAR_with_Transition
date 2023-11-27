# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sequence_length = 50
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return sequences
def train(file, model_name):
    # Load your dataset
    df = pd.read_csv(file)

    # Encode the 'activity' column
    le = LabelEncoder()
    df['activity'] = le.fit_transform(df['activity'])
    joblib.dump(le, 'label_encoder.pkl')

    # Create sequences for the accelerometer and gyroscope data
    X = df[['acc_z', 'acc_y', 'acc_x', 'gry_z', 'gry_y','gry_x']].values
    y = df['activity'].values
   

    if model_name == 'LSTM':
        X = create_sequences(X, sequence_length)
        y = y[sequence_length - 1:]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = to_categorical(y_train, num_classes=3)
        y_test = to_categorical(y_test, num_classes=3)
        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
        model.save("trained_model.h5")
        # Make predictions
        y_pred = model.predict(X_test)

        # You can use the predictions to identify the 'walkToRun' transition

        # Evaluate the model
        # loss, accuracy = model.evaluate(X_test, y_test)
        # print(f"Test loss: {loss}, Test accuracy: {accuracy}")
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(y_pred)
    # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        # acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        acc_series = pd.Series(acc, name='Accuracy')
        precision_series = pd.Series(precision, name='Precision')
        recall_series = pd.Series(recall, name='Recall')
        f1_series = pd.Series(f1, name='F1')

        df = pd.concat([acc_series, precision_series, recall_series, f1_series], axis=1)
        return df

    elif model_name == 'SVM':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        # y_train = to_categorical(y_train, num_classes=3)
        # y_test = to_categorical(y_test, num_classes=3)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = SVC(kernel='linear', C=1.0)
        model.fit(X_train, y_train)

        model_filename = 'svm_model.joblib'
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        acc_series = pd.Series(acc, name='Accuracy')
        precision_series = pd.Series(precision, name='Precision')
        recall_series = pd.Series(recall, name='Recall')
        f1_series = pd.Series(f1, name='F1')

        df = pd.concat([acc_series, precision_series, recall_series, f1_series], axis=1)
        return df
    elif model_name == 'DNN':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build the DNN model
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_split=0.2)

        # Evaluate the model
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        acc_series = pd.Series(acc, name='Accuracy')
        precision_series = pd.Series(precision, name='Precision')
        recall_series = pd.Series(recall, name='Recall')
        f1_series = pd.Series(f1, name='F1')

        df = pd.concat([acc_series, precision_series, recall_series, f1_series], axis=1)
        return df
# train('final.csv','DNN')