# IMDb Sentiment Analysis Using Bag-of-Words (BoW) Models
This project focuses on sentiment analysis using the IMDb movie reviews dataset. It implements and evaluates Bag-of-Words (BoW) representations through multiple modes: Binary, TF-IDF, Frequency, and Counts. The processed data is used to train a neural network for sentiment classification.
**Features**
- Preprocessing of text data: tokenization, lemmatization, and removal of stopwords and special characters.
- Implementation of four BoW models:
   - Binary
   - TF-IDF
   - Frequency
   - Counts
- Training of a neural network with TensorFlow/Keras for sentiment classification.
- Visualization of training and validation performance (accuracy and loss).
  
**Requirements**
- Install the following Python libraries to run the code:

  - TensorFlow
  - Numpy
  - Scikit-learn
  - NLTK
  - Matplotlib

# Workflow

**Text Preprocessing**
- Normalization: Removes special characters, punctuation, and non-ASCII characters, and converts text to lowercase.
- Tokenization: Splits text into words.
- Stopword Removal: Removes commonly used words that don't contribute to sentiment analysis.
- Stemming/Lemmatization: Reduces words to their root forms for consistent analysis.

**BoW Models**
- Binary: Encodes presence/absence of words in reviews.
- TF-IDF: Weights words by their frequency and importance.
- Frequency: Captures the count of each word in a review.
- Counts: Similar to frequency, provides raw word counts.

**Neural Network**
- Input: BoW vectors of size 10,000.
- Architecture: Sequential model with dense layers and ReLU activation.
- Output: Sigmoid activation for binary classification (positive/negative sentiment).
- Loss Function: Binary cross-entropy.

**Model Training and Evaluation**
- Training data is split into training and validation sets.
- Metrics: Binary accuracy, training/validation loss.
- Visualizations: Training progress is plotted for both accuracy and loss.
