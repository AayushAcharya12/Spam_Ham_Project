import streamlit as st
import joblib

#Load vecotrizer and required models 
vectorizer = joblib.load("model/vectorizer.pkl")
nb_model = joblib.load("model/naive_bayes.pkl")
lr_model = joblib.load("model/logistic_regression.pkl")

st.title(" Spam Detection App (ML Models)")

st.write("Enter a message and choose a model")

#Input area to enter a message
text = st.text_area("Enter your message here")

# Select The Model
model_choice = st.selectbox(
    "Choose Model",
    ["Naive Bayes", "Logistic Regression"]
)

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a message")
    else:
        # Convert text  into vector
        X = vectorizer.transform([text])

        # Prediction
        if model_choice == "Naive Bayes":
            pred = nb_model.predict(X)
            model_used = "Naive Bayes"

        else:
            pred = lr_model.predict(X)
            model_used = "Logistic Regression"

        # Output
        if pred[0] == True:
            st.error(f"Spam detected (using {model_used})")
        else:
            st.success(f"Ham message (using {model_used})")