# detect_emotion.py

import joblib
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# Load the trained models and vectorizer
model = joblib.load('model2.pkl')
svm_model = joblib.load('model1.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Emotion detection function
def detect_emotion(text):
    # preprocessed_text = ' '.join([stemmer.stem(word.lower()) for word in nltk.word_tokenize(text) if word.isalpha() and word.lower() not in stop_words])

    # Ensure the vectorizer is fitted before transforming
    preprocessed_data = vectorizer.transform([text])

    # Your code here to detect emotion in the input text
    predicted_label1 = model.predict(preprocessed_data)[0]
    predicted_label2 = svm_model.predict(preprocessed_data)[0]
    # predicted_label3 = rf_model.predict(preprocessed_data)[0]

    # Combine predictions
    predictions = [predicted_label1, predicted_label2]

    # Count the occurrences of each predicted label
    label_counts = Counter(predictions)

    # Find the label with the maximum count
    most_common_label = label_counts.most_common(1)[0][0]

    # Define emoji mapping
    emoji_mapping = {'sadness': "😞", 'surprise': "😮", 'love':'🥰', 'happy':"😄", 'anger':"😡", 'fear':"😱"}

    return most_common_label + emoji_mapping[most_common_label], predictions
