import streamlit as slt
import pickle
import nltk
import string
p = string.punctuation
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

from nltk.corpus import stopwords


with open('random_forest.pkl', 'rb') as file:
    forest_classifier = pickle.load(file)
with open('tf_vector.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)


slt.title('Random Forest Classifier for Spam/Ham Classification')
user_input = slt.text_area("Enter your SMS to know if it is HAM or SPAM")


def sms_preprocess(sms):
    sms = sms.lower()
    sms = nltk.word_tokenize(sms)
    l = []
    for i in sms:
        if i.isalnum():
            l.append(i)
    stop_words = stopwords.words('english')
    second_l = []
    for j in l:
        if j not in stop_words:
            second_l.append(j)
    third_l = []
    for k in second_l:
        if k not in p:
            third_l.append(k)
    fourth_l = []
    for m in third_l:
        stemmed_words = stemmer.stem(m)
        fourth_l.append(stemmed_words)

    return " ".join(fourth_l)




if slt.button('Click to know if it is Ham or Spam'):
    preprocessed = sms_preprocess(user_input)
    vectorized = tfidf_vectorizer.transform(preprocessed)
    prediction = forest_classifier.predict(vectorized)[0]
    if prediction == 0:
        slt.header("HAM!! Read it.")
    elif prediction == 1:
        slt.header("SPAM!! Ignore it.")




