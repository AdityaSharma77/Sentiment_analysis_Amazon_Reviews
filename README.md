# Sentiment_analysis_Amazon_Reviews

This repository contains code for sentiment analysis using different machine learning algorithms: Naive Bayes, Support Vector Machines (SVM), and Recurrent Neural Networks (RNNs) with LSTM.

## Dataset

The sentiment analysis is performed on the 'amazon_reviews.csv' dataset, which contains customer reviews from Amazon. The dataset is preprocessed and cleaned to remove any rows with missing values in the 'reviewText' and 'overall' columns.

## Naive Bayes

The Naive Bayes algorithm is implemented using the `MultinomialNB` class from the `sklearn.naive_bayes` module. The text data is converted into numerical feature vectors using the `CountVectorizer` class from the `sklearn.feature_extraction.text` module. The trained model is evaluated using the accuracy metric.

## Support Vector Machines (SVM)

The SVM algorithm is implemented using the `SVC` class from the `sklearn.svm` module. The text data is converted into numerical feature vectors using the `TfidfVectorizer` class from the `sklearn.feature_extraction.text` module. The trained model is evaluated using the accuracy metric.

## Recurrent Neural Networks (RNNs) with LSTM

The RNN model with LSTM is implemented using the Keras API with TensorFlow backend. The text data is tokenized and converted into sequences using the `Tokenizer` class from the `tensorflow.keras.preprocessing.text` module. The sequences are then padded to ensure equal lengths using the `pad_sequences` function from the `tensorflow.keras.preprocessing.sequence` module. The RNN model consists of an embedding layer, LSTM layer, and a dense layer. The model is compiled with the binary cross-entropy loss function and trained using the Adam optimizer. The model is evaluated using accuracy as the metric.

## Evaluation Metrics

For each algorithm, the following evaluation metrics are calculated:
- Accuracy: the proportion of correctly predicted sentiments.
- Precision: the ability of the model to identify positive instances correctly.
- Recall: the ability of the model to correctly detect positive instances.
- F1-score: the harmonic mean of precision and recall.

## Comparison of Evaluation Metrics

A comparison of the evaluation metrics (accuracy, precision, recall, and F1-score) is visualized using bar charts. The charts illustrate the performance of each algorithm.

## Usage

1. Clone the repository:
   ```shell
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis

pip install -r requirements.txt

python sentiment_analysis.py

Requirements
Python 3.x
pandas
scikit-learn
TensorFlow
Keras
matplotlib

License
This project is licensed under the MIT License. See the LICENSE file for details.
