import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from train import train
from predict import predict

st.set_page_config(layout="wide")
st.title("Model Training and Prediction")

selected_tab = st.sidebar.radio("Select an action:", ["Train Model", "Predict Data"])

options = ["LSTM", "SVM", "DNN"]

# Create the dropdown in the sidebar
selected_model = st.sidebar.selectbox("Select a model:", options)

# Display the selected option
st.sidebar.write(f"You selected {selected_model} model...")

if selected_tab == "Train Model":
    st.sidebar.subheader("Upload a CSV for Training")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:

        with st.spinner("Training the model..."):
            res_df = train(uploaded_file,selected_model)

        st.dataframe(res_df)
        st.sidebar.success("Model trained successfully!")

if selected_tab == "Predict Data":
    st.sidebar.subheader("Upload a CSV for Prediction")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        pred_df = predict(uploaded_file)
        activity_counts = pred_df['pred_activity'].value_counts()
        ordered_activities = activity_counts.index.tolist()
        st.write("## Transition: ",ordered_activities[1])

        tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        tab2.dataframe(pred_df.iloc[:,1:],width=1200)
        col1, col2 = tab1.columns(2)     
        with col1:
            st.line_chart(pred_df, x="seconds_elapsed", y="acc_x", color="pred_activity",use_container_width=True)
            st.line_chart(pred_df, x="seconds_elapsed", y="acc_y", color="pred_activity",use_container_width=True)
            st.line_chart(pred_df, x="seconds_elapsed", y="acc_z", color="pred_activity",use_container_width=True)

        with col2:
            st.line_chart(pred_df, x="seconds_elapsed", y="gry_x", color="pred_activity",use_container_width=True)
            st.line_chart(pred_df, x="seconds_elapsed", y="gry_y", color="pred_activity",use_container_width=True)
            st.line_chart(pred_df, x="seconds_elapsed", y="gry_z", color="pred_activity",use_container_width=True)


