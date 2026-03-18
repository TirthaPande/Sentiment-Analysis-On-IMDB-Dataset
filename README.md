# Sentiment-Analysis-On-IMDB-Dataset
NLP Mini Project: Text Processing and Sentiment Analysis
Overview
This project focuses on performing sentiment analysis on movie reviews using various machine learning models. It covers essential Natural Language Processing (NLP) steps, including text preprocessing, feature extraction (TF-IDF), model training, evaluation, and hyperparameter tuning.

Dataset
The project utilizes the IMDB Dataset.csv, which contains movie reviews labeled as either 'positive' or 'negative' sentiment.

Project Structure
The notebook (C4_56_Tirtha_Pande_NLP_LAB8.ipynb) is structured into the following key sections:

Library Imports and NLTK Downloads: Necessary libraries like pandas, nltk, sklearn, and seaborn are imported. NLTK data (punkt, stopwords, wordnet) is downloaded.
Data Loading: The IMDB Dataset.csv is loaded into a pandas DataFrame.
Text Preprocessing: A preprocess function is defined to clean the text data, including:
Lowercasing
Tokenization
Removal of non-alphabetic tokens
Removal of stopwords
Lemmatization A new column clean_review is added to the DataFrame.
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert the cleaned text reviews into numerical features. TfidfVectorizer with max_features=5000 is applied.
Data Splitting: The dataset is split into training and testing sets using train_test_split (80% training, 20% testing).
Model Training and Evaluation (Baseline):
A MultinomialNB model is trained and evaluated.
Accuracy, F1-score, and a classification report are generated.
A confusion matrix is visualized using seaborn.heatmap.
Prediction on New Text: Examples demonstrate how to predict sentiment for new, unseen movie review texts.
Comparative Model Evaluation: Several classification models are trained and evaluated for comparison:
Multinomial Naive Bayes
Logistic Regression
Support Vector Machine (SVC)
Random Forest Classifier
Gradient Boosting Classifier Results (Accuracy and F1 Score) are stored in a DataFrame and visualized using a bar plot.
Hyperparameter Tuning: GridSearchCV is used to find optimal hyperparameters for:
Logistic Regression: Tuning the C parameter.
Support Vector Machine: Tuning C and kernel parameters. The performance of the tuned models is then reported.
Key Findings
Initial evaluations showed Logistic Regression and Support Vector Machine performing best among the untuned models.
Hyperparameter tuning further optimized the Logistic Regression and Support Vector Machine models, leading to slight improvements in accuracy and F1-score.
How to Run
Open in Google Colab: Upload the IMDB Dataset.csv to your Google Drive or Colab session. Ensure the notebook is also accessible in Colab.
Mount Google Drive: The notebook includes code to mount Google Drive for accessing the dataset if stored there.
Install Dependencies: All necessary libraries are standard in Colab. NLTK data is downloaded programmatically.
Execute Cells: Run the cells sequentially to perform data loading, preprocessing, model training, evaluation, and prediction.
Requirements
Python 3.x
pandas
scikit-learn
nltk
seaborn
matplotlib
Files
C4_56_Tirtha_Pande_NLP_LAB8.ipynb: The main Colab notebook containing all the code and analysis.
IMDB Dataset.csv: The dataset used for sentiment analysis.
C4_56_Tirtha_Pande_NLP_LAB8.pkl: A pickled version of the notebook (generated at the end of the notebook).
