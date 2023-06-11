import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

import chardet

with open('train.csv', 'r', encoding='utf-8', errors='replace') as f:
    train_df = pd.read_csv(f)

with open('test.csv', 'r', encoding='utf-8', errors='replace') as f:
    test_df = pd.read_csv(f)



# Preprocessing - Vectorization of text
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
train_vectors = vectorizer.fit_transform(train_df['OriginalTweet'])
test_vectors = vectorizer.transform(test_df['OriginalTweet'])

# Encoding Labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_df['Sentiment'])
test_labels = encoder.transform(test_df['Sentiment'])

# Training the classifier
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, train_labels)


# Save the trained model, vectorizer and encoder
pickle.dump(classifier_linear, open("svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))


# Making predictions
prediction_linear = classifier_linear.predict(test_vectors)

# Printing the classification report
print(classification_report(test_labels, prediction_linear, target_names=encoder.classes_))
