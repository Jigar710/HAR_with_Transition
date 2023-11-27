from keras.models import load_model
import pandas as pd
import numpy as np
import joblib
from keras.utils import to_categorical
sequence_length = 50

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return sequences

def predict(file, model_name):
    model = load_model("trained_model.h5")
    test_df = pd.read_csv(file)
    le = joblib.load('label_encoder.pkl')
    # test_df['activity'] = le.transform(test_df['activity'])
    X_test = test_df[['acc_z', 'acc_y', 'acc_x', 'gry_z', 'gry_y','gry_x']].values
    X_test = create_sequences(X_test, sequence_length)
    # y_test = test_df['activity']
    # y_test = y_test[sequence_length - 1:]
    X_test = np.array(X_test)
    # y_test = to_categorical(y_test, num_classes=3)
    
    # loss, accuracy = model.evaluate(X_test, y_test)
    # print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    predictions = model.predict(X_test)
    pred = []
    for i in predictions:
        pred.append(np.argmax(i))
    labeled_pred = le.inverse_transform(np.array(pred))
    templst = [labeled_pred[1] for i in range(sequence_length-1)]
    print(templst)
    labeled_pred = templst.extend(list(labeled_pred))
    test_df['pred_activity'] = templst
    # con_df = test_df[['activity','pred_activity']]
    # con_df.to_csv('pred_test.csv', index=False)
    first_element = test_df.at[0, "seconds_elapsed"]
    test_df["seconds_elapsed"] = test_df["seconds_elapsed"] - first_element
    plot_df = test_df[['seconds_elapsed','acc_z', 'acc_y', 'acc_x', 'gry_z', 'gry_y','gry_x','pred_activity']]
    return test_df


# predict("test01.csv","LSTM")