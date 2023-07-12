```markdown
# Sentiment Analysis with SVM Classifier

This repository contains code to train and use an SVM classifier with different kernels to perform sentiment analysis on text.

## Prerequisites

The code is written in Python. You need to have Python 3.x installed. Also, make sure the following libraries are installed:

- pandas
- numpy
- sklearn
- pickle
- chardet

You can install these with pip:
```shell
pip install pandas numpy sklearn pickle-mixin chardet
```

## Training the Classifier

This repository contains scripts to train the SVM classifier using three different kernels: linear (`train_model_linear.py`), polynomial (`train_model_poly.py`), and radial basis function (RBF) (`train_model_rbf.py`). Each script performs the following steps:

1. Load training and testing data from CSV files.
2. Preprocess the data (including vectorization of text).
3. Train the SVM classifier with the respective kernel.
4. Predict the sentiments for the test data.
5. Print a classification report.
6. Save the trained model, vectorizer, and label encoder to pickle files.

You can run these scripts with the command:
```shell
python train_model_<kernel>.py
```
Replace `<kernel>` with `linear`, `poly`, or `rbf` depending on which kernel you want to use.

## Using the Classifier

After training the classifier, you can use the `get_sentiment.py` script to classify the sentiment of a text string. This script loads the trained model, vectorizer, and label encoder from the pickle files, and then uses them to classify the sentiment of an input string.

You can use this script as follows:
```python
import get_sentiment

text = "Your input text here"
print(get_sentiment.get_sentiment(text))
```
This will print the sentiment of the input text.

## Note

The classification model, vectorizer, and label encoder are all saved as pickle files after training, and these are loaded each time the `get_sentiment.py` script is run. This means that you don't have to retrain the classifier every time you want to classify a text string. If you want to train a new classifier (e.g., if you have new training data), just delete the pickle files and run the respective training script again.
```

Remember to close the shell and python code blocks properly in the actual README file.
