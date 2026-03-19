import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Download required NLTK data
# -------------------------------
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk()

# -------------------------------
# Load model and vectorizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer_loaded = pickle.load(f)

    with open('best_LOG_REG__model.pkl', 'rb') as f:
        model_loaded = pickle.load(f)

    return vectorizer_loaded, model_loaded

vectorizer, model = load_artifacts()

# -------------------------------
# Preprocessing setup
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()

    # safer than word_tokenize (avoids deployment errors)
    tokens = text.split()

    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Sentiment Predictor", layout="centered")

st.title("🎬 Movie Review Sentiment Predictor")
st.write("Enter a movie review to predict whether it's positive or negative.")

user_input = st.text_area("Your Movie Review:")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        try:
            cleaned_text = preprocess(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            sentiment = prediction[0]

            if sentiment == 'positive':
                st.success(f"Predicted Sentiment: {sentiment.upper()} 🎉")
            else:
                st.error(f"Predicted Sentiment: {sentiment.upper()} 😠")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a review.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.info("This app uses a TF-IDF + Support Vector Machine model.")
