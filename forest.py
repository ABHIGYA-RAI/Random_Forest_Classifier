import streamlit as slt
import pickle
import numpy as np

slt.title('Random Forest Classifier for Spam/Ham Classification')
user_input = slt.text_area("Enter your SMS to know if it is HAM or SPAM")

if slt.button('Click to know if it is Ham or Spam'):
    with open('random_forest.pkl', 'rb') as file:
        forest_classifier = pickle.load(file)
    with open('tf_vector.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    arr_input = np.array([user_input.split(',')],dtype=float)
    prediction = forest_classifier.predict(arr_input)
    if prediction[0] == 0:
        slt.header("HAM!! Read it.")
    elif prediction[0] == 1:
        slt.header("SPAM!! Ignore it.")




