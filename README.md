
# Sentiment Analysis using TF-IDF and Machine Learning Models

This project performs **sentiment analysis** on the IMDB dataset using text classification techniques. The dataset consists of movie reviews labeled as either **positive** or **negative**, and the goal is to classify these reviews based on their sentiment.

## Project Overview

The project uses **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text data into numerical vectors. Several machine learning models, including **Decision Trees** and **Random Forest**, are trained to classify the sentiment of the reviews.

## Key Features

- **Text Preprocessing**: 
  - Expanding contractions (e.g., "don't" → "do not").
  - Removing HTML tags, emojis, URLs, and punctuation.
  - Lemmatization and stopword removal.
  
- **TF-IDF Vectorization**: 
  - Transform the preprocessed text data into numerical vectors using **TfidfVectorizer**.

- **Models Implemented**:
  - **Decision Tree Classifier**: A simple model that builds a tree-like structure to classify reviews.
  - **Random Forest Classifier**: An ensemble method that combines multiple decision trees to improve performance and reduce overfitting.

- **Evaluation**: 
  - Split the dataset into training and testing sets using **train_test_split**.
  - Use **accuracy score** to evaluate the performance of the trained models.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   ```

2. **Install required dependencies**:
   Make sure you have Python 3 installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional NLTK resources**:
   Some NLTK resources need to be downloaded. You can add the following to your code or download them manually:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

1. **Data Preprocessing**:
   The raw text is cleaned and transformed into numerical vectors:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   tfidf_vectorizer = TfidfVectorizer(max_features=10000)
   x_train_encoded = tfidf_vectorizer.fit_transform(x_train)
   ```

2. **Train the Model**:
   You can train different models such as Decision Trees or Random Forest:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
   dt_classifier.fit(x_train_encoded, y_train)
   ```

3. **Evaluate the Model**:
   After training, use the model to predict sentiments and evaluate its accuracy:
   ```python
   from sklearn.metrics import accuracy_score
   y_pred = dt_classifier.predict(x_test_encoded)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy}")
   ```

4. **Random Forest Example**:
   Train and evaluate a Random Forest Classifier:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
   rf_classifier.fit(x_train_encoded, y_train)
   ```

## Example Predictions

You can test the trained model with a few example reviews:

```python
example_reviews = ["The movie was fantastic!", "I didn't like the movie at all."]
example_encoded = tfidf_vectorizer.transform(example_reviews)
predictions = rf_classifier.predict(example_encoded)
decoded_labels = label_encode.inverse_transform(predictions)
print(decoded_labels)
```

## Dataset

This project uses the **IMDB Movie Review Dataset**, which contains 50,000 reviews labeled as either positive or negative. The dataset can be downloaded from [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Dependencies

- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- BeautifulSoup4
- Contractions
- Matplotlib (for visualizations)
- Seaborn (for visualizations)

## Project Structure

```
.
├── data/                # Folder containing dataset (IMDB-Dataset.csv)
├── notebooks/           # Jupyter notebooks for experimentation
├── models/              # Trained models (if saved)
├── README.md            # Project documentation
├── requirements.txt     # Dependencies for the project
└── src/                 # Main source code for the project
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or contributions, please reach out via [your-email@example.com].
