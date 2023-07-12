import pickle
import numpy as np


def get_sentiment(input_text):
    # Load the trained model, vectorizer, and label encoder
    classifier = pickle.load(open("svm_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))

    # Vectorize the input text
    input_vector = vectorizer.transform([input_text])

    # Predict sentiment of the input text
    prediction = classifier.predict(input_vector)

    # Decode the sentiment label
    sentiment_label = encoder.inverse_transform(prediction)

    return sentiment_label[0]


# Example usage:
input_text = "fuck"
print(f"Sentiment: {get_sentiment(input_text)}")
