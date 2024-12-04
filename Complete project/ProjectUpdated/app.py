import streamlit as st
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page icon and title
st.set_page_config(
    page_title="Email Classification",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed")

# lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Dictionary for expanding contractions

# Dictionary for expanding contractions
contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "won’t": "will not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(text, contractions):
    text = text.replace("’", "'")
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    text = str(text).strip().lower()
    text = expand_contractions(text, contractions)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("\n", " ").replace("\t", " ")
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Load all your models
models = {
    "DecisionTreeClassifier": joblib.load("models/best_DecisionTreeClassifier_model.h5"),
    "Support Vector Machine Classifier (SVM)": joblib.load("models/best_SVM_model.h5"),
    "Random Forest Classifier": joblib.load("models/best_RandomForestClassifier_model.h5"),
    "K-Nearest Neighbors (KNN)": joblib.load("models/best_KNN_model.h5"),
    "Gradient Boosting Classifier": joblib.load("models/best_GradientBoostingClassifier_model.h5"),
    "Tuned DecisionTreeClassifier": joblib.load("models/best_DecisionTreeClassifier_model_tuned.h5"),
    "Tuned Support Vector Machine Classifier (SVM)": joblib.load("models/best_SVM_model_tuned.h5"),
    "Tuned Random Forest Classifier": joblib.load("models/best_RandomForestClassifier_model_tuned.h5"),
    "Tuned K-Nearest Neighbors (KNN)": joblib.load("models/best_KNN_model_tuned.h5"),
    "Tuned Gradient Boosting Classifier": joblib.load("models/best_GradientBoostingClassifier_model_tuned.h5"),
}
# Dropdown to select the model
model_name = st.selectbox("Select a Model for Prediction", list(models.keys()))

# Load the selected model
model = models[model_name]

# load vectorizer and svd
vectorizer = joblib.load('tfidf_vectorizer.pkl')
svd = joblib.load('svd_model.pkl')


# CSS for styling
st.markdown("""
    <style>
    .main {
        background: #688f9c;
        padding: 4 rem;
    }
    h1 {
        background-color: #4c5caf;
        color: white;
        text-align: center;
        font-size: 36px;
        border-radius: 60px;
        padding: 1 em;
    }
    .stButton>button {
        display: block;
        margin: 0 auto;
        background-color: #17178a;
        color: white;
        padding: 0.5em 1em;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        font-size: 1em;
    }
    .stTextArea textarea {
        border-radius: 4px;
        border: 5px solid #ccc;
    }
            
    .stTextArea label {
        color: white
    }
            

    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown(f'<h1>{"Email Classification"}</h1>', unsafe_allow_html=True)
st.write("\n\n")

# Text input
user_input = st.text_area("Enter the email text here", height=200)


# Classify button
if st.button("Classify", key="classify_button"):
    if not user_input:
        st.markdown("<h2 style='color:red;'>Please write an email.</h2>", unsafe_allow_html=True)
    else:
        # Preprocess the input
        processed_text = preprocess_text(user_input)
        
        # Vectorize the input
        vectorized_text = vectorizer.transform([processed_text]).toarray()

        # Reduce dimensions with SVD
        reduced_text = svd.transform(vectorized_text)
        
        # Predict the label
        prediction = model.predict(reduced_text)
        
        # Map the prediction to label
        label_mapping = {
            0: ('Human Non-Phishing', 'white'),
            1: ('LLM Phishing', 'blue'),
            2: ('Human Phishing', 'blue'),
            3: ('LLM Non-Phishing', 'white')
        }
        prediction_label, color = label_mapping[prediction[0]]

        # Display the result
        st.markdown(f"<h2 style='color:{color};'>The email is classified as: <strong>{prediction_label}</strong></h2>", unsafe_allow_html=True)
        st.balloons()

