from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
with open('train.csv', 'r', encoding='utf-8', errors='replace') as f:
    train_df = pd.read_csv(f)

with open('test.csv', 'r', encoding='utf-8', errors='replace') as f:
    test_df = pd.read_csv(f)

# Preprocessing - Vectorization of text
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
train_vectors = vectorizer.fit_transform(train_df['OriginalTweet'])
test_vectors = vectorizer.transform(test_df['OriginalTweet'])

# Saving vectorizer
with open('vectorizer_rbf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Encoding Labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_df['Sentiment'])
test_labels = encoder.transform(test_df['Sentiment'])

# Saving encoder
with open('encoder_rbf.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Training the classifier with RBF kernel
classifier_rbf = svm.SVC(kernel='rbf')
classifier_rbf.fit(train_vectors, train_labels)

# Saving the model
with open('svm_rbf_model.pkl', 'wb') as f:
    pickle.dump(classifier_rbf, f)

# Making predictions
prediction_rbf = classifier_rbf.predict(test_vectors)

# Printing the classification report
print(classification_report(test_labels, prediction_rbf, target_names=encoder.classes_))

# Generating confusion matrix
conf_matrix = confusion_matrix(test_labels, prediction_rbf)

# Visualizing the confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
