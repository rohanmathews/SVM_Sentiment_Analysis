import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel
from sklearn.metrics import classification_report, confusion_matrix

# Load the datasets
with open('train.csv', 'r', encoding='utf-8', errors='replace') as f:
    train_df = pd.read_csv(f)

with open('test.csv', 'r', encoding='utf-8', errors='replace') as f:
    test_df = pd.read_csv(f)

# Preprocessing - Vectorization of text
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
train_vectors = vectorizer.fit_transform(train_df['OriginalTweet'])
test_vectors = vectorizer.transform(test_df['OriginalTweet'])

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer_hybrid.pkl", "wb"))

# Encoding Labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_df['Sentiment'])
test_labels = encoder.transform(test_df['Sentiment'])

# Save encoder
pickle.dump(encoder, open("encoder_hybrid.pkl", "wb"))

# Hybrid kernel
def hybrid_kernel(X, Y):
    return polynomial_kernel(X, Y, degree=3) + rbf_kernel(X, Y, gamma=0.5) + linear_kernel(X, Y)

# Training the classifier
classifier_hybrid = svm.SVC(kernel=hybrid_kernel)
classifier_hybrid.fit(train_vectors, train_labels)

# Save the model
pickle.dump(classifier_hybrid, open("svm_model_hybrid.pkl", "wb"))

# Making predictions
prediction_hybrid = classifier_hybrid.predict(test_vectors)

# Printing the classification report
print(classification_report(test_labels, prediction_hybrid, target_names=encoder.classes_))

# Printing the confusion matrix
print(confusion_matrix(test_labels, prediction_hybrid))
