# 🎬 Sentiment Analysis on IMDB Dataset

🧠 NLP Mini Project: Text Processing and Sentiment Analysis

---

## 📌 Overview

This project performs **sentiment analysis on movie reviews** using various machine learning models. It demonstrates core Natural Language Processing (NLP) techniques including:

- Text preprocessing  
- Feature extraction (TF-IDF)  
- Model training and evaluation  
- Hyperparameter tuning  

---

## 📊 Dataset

- **File:** `IMDB Dataset.csv`  
- Contains movie reviews labeled as:
  - ✅ Positive  
  - ❌ Negative  

---

## 📂 Project Structure

📁 Sentiment-Analysis-On-IMDB-Dataset
│── 📄 C4_56_Tirtha_Pande_NLP_LAB8.ipynb # Main notebook
│── 📄 IMDB Dataset.csv # Dataset
│── 📄 C4_56_Tirtha_Pande_NLP_LAB8.pkl # Serialized notebook
│── 📄 README.md # Documentation



---

## ⚙️ Workflow

### 🔹 1. Library Imports & Setup
- Libraries used: `pandas`, `nltk`, `sklearn`, `seaborn`, `matplotlib`
- NLTK resources downloaded:
  - `punkt`
  - `stopwords`
  - `wordnet`

---

### 🔹 2. Data Loading
- Dataset loaded into a pandas DataFrame

---

### 🔹 3. Text Preprocessing

A custom `preprocess()` function performs:

- Lowercasing  
- Tokenization  
- Removing non-alphabetic words  
- Stopword removal  
- Lemmatization  

➡️ New column: `clean_review`

---

### 🔹 4. Feature Extraction

- TF-IDF Vectorization  
- `TfidfVectorizer(max_features=5000)`

---

### 🔹 5. Train-Test Split

- 80% Training  
- 20% Testing  

---

### 🔹 6. Model Training & Evaluation

Baseline Model:
- ✅ Multinomial Naive Bayes  

Evaluation Metrics:
- Accuracy  
- F1 Score  
- Classification Report  
- Confusion Matrix (Seaborn Heatmap)

---

### 🔹 7. Prediction on New Data

- Predict sentiment for unseen reviews  

---

### 🔹 8. Comparative Model Analysis

Models used:

- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gradient Boosting Classifier  

📊 Results visualized using bar plots

---

### 🔹 9. Hyperparameter Tuning

Using `GridSearchCV`:

- Logistic Regression → tuning `C`  
- SVM → tuning `C` and `kernel`  

---

## 📈 Key Findings

- Logistic Regression and SVM performed best initially  
- Hyperparameter tuning improved performance further  
- TF-IDF proved effective for text representation  

---

## 🚀 How to Run

### ▶️ Option 1: Google Colab (Recommended)

1. Open Google Colab  
2. Upload:
   - Notebook (`.ipynb`)
   - Dataset (`IMDB Dataset.csv`)  
3. Run all cells  

---

### ▶️ Option 2: Local Setup

```bash
pip install pandas scikit-learn nltk seaborn matplotlib
jupyter notebook

