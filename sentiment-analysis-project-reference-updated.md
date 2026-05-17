# Sentiment Analysis of Text Reviews Using Machine Learning and Visualization

## Project Description

This project builds a sentiment analysis system that reads user-generated text such as movie reviews, product reviews, or tweets and classifies each input as positive or negative. The system first cleans and preprocesses the text, converts it into numerical form using TF-IDF, trains machine learning classifiers, and then presents the final results through evaluation metrics and visualizations such as sentiment distribution, word frequency, model comparison, and confusion matrix plots.[cite:28][cite:36][cite:49]

The main goal is to show how Natural Language Processing and Machine Learning can be combined to automatically understand opinion from text data. This is a practical and defendable college project because every stage—preprocessing, vectorization, classification, evaluation, and visualization—can be explained clearly and implemented using common Python libraries.[cite:35][cite:37]

## Dataset Used

The recommended dataset for this project is the **IMDb movie review dataset**, because it is one of the standard benchmark datasets for binary sentiment classification and contains positive and negative movie reviews.[cite:79][cite:78]

A commonly used CSV version of this dataset contains **50,000 labeled movie reviews**, with **25,000 positive** and **25,000 negative** samples.[cite:78][cite:80]

The original Large Movie Review Dataset also provides a standard split of **25,000 training reviews** and **25,000 testing reviews**, with labels defined as `0 = negative` and `1 = positive` in common processed versions.[cite:79]

For this project, the dataset should be stored in the codebase as a separate file such as:

```text
data/IMDB Dataset.csv
```

The dataset should **not** be hardcoded inside Python files. Instead, the code should load it from the `data/` folder using pandas. This is the most practical setup for a college project because it keeps the code clean and makes the project easy to run on another system.[cite:73][cite:81]

If the dataset is not included directly in the repository, the README should clearly mention the download source and tell the user to place the file inside the `data/` folder before running the training script.[cite:73][cite:84]

## Architecture

At a high level, the system follows a linear NLP pipeline where raw text enters the system, gets preprocessed, is transformed into TF-IDF features, passed into trained classifiers, and then the results are used for prediction and visualization. This pipeline structure is directly aligned with standard scikit-learn text-processing architecture, where text is vectorized and then classified using a pipeline of feature extraction and classifier stages.[cite:35][cite:44]

### Architecture Flow

User Review / Dataset  
↓  
Text Preprocessing  
↓  
Feature Extraction using TF-IDF  
↓  
Model Training / Model Testing  
↓  
Sentiment Prediction  
↓  
Evaluation + Visualization

The architecture can be explained in four blocks:[cite:35][cite:36]

- **Input Layer:** Dataset of reviews or tweets.[cite:28][cite:36]
- **Processing Layer:** Cleaning, token handling, stopword removal, and normalization.[cite:28][cite:48]
- **ML Layer:** TF-IDF vectorizer plus classifiers like Logistic Regression, Naive Bayes, and Linear SVM.[cite:35][cite:45]
- **Output Layer:** Predicted sentiment, charts, metrics, and comparison results.[cite:36][cite:37]

## Project Flow

The project flow begins by loading a labeled dataset that contains text and sentiment labels. After that, the data is cleaned by converting text to lowercase, removing punctuation, special symbols, extra spaces, and optionally stopwords or stemmed forms before feature extraction begins.[cite:28][cite:36][cite:48]

Next, the cleaned text is converted into numerical vectors using TF-IDF, which assigns higher importance to words that are frequent in a document but less common across all documents. These feature vectors are then split into training and testing sets, the models are trained on the training data, predictions are generated on test data, and the results are visualized using graphs and performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix.[cite:49][cite:35][cite:36]

### Step-by-Step Flow

1. Load dataset from `data/IMDB Dataset.csv`.
2. Clean and normalize text.
3. Convert text to TF-IDF vectors.
4. Split into train and test data.
5. Train ML models.
6. Predict sentiment on test data.
7. Compare model performance.
8. Visualize results.
9. Allow prediction for new custom text input.

## High Level Design

The high level design divides the project into major functional modules so the system is easy to build and explain. Each module performs one major task and passes its output to the next module in sequence, which is the common pattern for a traditional ML-based text classification project.[cite:35][cite:36]

### 1. Data Input Module
This module loads the sentiment dataset from a CSV file or similar structured source. The dataset generally contains two columns: one for text and one for sentiment label such as positive or negative.[cite:28][cite:36]

### 2. Preprocessing Module
This module cleans raw text to improve model quality. Typical steps include lowercase conversion, punctuation removal, HTML or noise removal if needed, stopword removal, and optional stemming or lemmatization.[cite:36][cite:48]

### 3. Feature Extraction Module
This module converts cleaned text into machine-readable numeric vectors using TF-IDF. TF-IDF is commonly used in sentiment analysis because it highlights words that are important in a given review while reducing the influence of overly common words.[cite:49][cite:35]

### 4. Model Training Module
This module trains machine learning models on the TF-IDF vectors. A standard setup is to use Logistic Regression as the main classifier, Naive Bayes as a baseline, and optionally Linear SVM for comparison.[cite:45][cite:35][cite:36]

### 5. Evaluation Module
This module checks how well the model performs using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. These metrics are widely used in text classification to compare classifiers and understand classification quality.[cite:36][cite:37]

### 6. Visualization Module
This module generates charts such as class distribution, most frequent words, confusion matrix heatmap, and model comparison graph. Visualization helps make the output easy to understand and improves presentation quality in reports and demos.[cite:37][cite:36]

### 7. Prediction Module
This module accepts a new review entered by the user, applies the same preprocessing and TF-IDF transformation, and then predicts the sentiment using the trained model. Scikit-learn pipelines support this by applying the same feature-transformation and classification logic to new text inputs.[cite:35][cite:44]

## Low Level Design

The low level design explains what happens inside each module and how data moves in code.

### Input Details
- Read CSV using pandas.
- Select text column and label column.
- Remove null rows and duplicates before training.

### Preprocessing Details
For each text record:
- Convert text to lowercase.
- Remove punctuation, numbers, and special characters.
- Remove extra spaces.
- Remove stopwords.
- Optionally apply stemming or lemmatization.[cite:28][cite:48]

### Feature Extraction Details
- Use `TfidfVectorizer`.[cite:35]
- Common settings can include `max_features=5000` and `ngram_range=(1,2)` for unigrams and bigrams, which are also used in public TF-IDF sentiment-analysis examples.[cite:45]
- Fit the vectorizer on training text only.
- Transform training and test text into sparse TF-IDF matrices.[cite:35]

### Training Details
- Train `LogisticRegression()` on TF-IDF features.
- Train `MultinomialNB()` as a baseline model.
- Optionally train `LinearSVC()` for comparison.
- Store trained model and vectorizer using pickle or joblib for reuse.

A scikit-learn `Pipeline` can combine vectorization and classification so that training and prediction happen in a single reusable workflow.[cite:35][cite:44]

### Evaluation Details
- Predict sentiment on test set.
- Generate `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.
- Build a confusion matrix.
- Produce a classification report for each model.[cite:36][cite:37]

### Visualization Details
- Bar chart of positive vs negative records.
- Word cloud of important words.
- Heatmap of confusion matrix.
- Bar chart comparing model accuracies.
- Optional top TF-IDF terms per class.

## Tech Stack

The project can be implemented fully in Python using a lightweight data-science stack. Scikit-learn provides the text vectorization, pipelines, and classifiers needed for TF-IDF-based sentiment analysis.[cite:35][cite:30]

| Layer | Technology |
|---|---|
| Programming language | Python [cite:35] |
| Data handling | Pandas, NumPy |
| NLP preprocessing | NLTK or Python `re` for cleaning [cite:28][cite:48] |
| Feature extraction | `TfidfVectorizer` from scikit-learn [cite:35][cite:49] |
| Models | Logistic Regression, Multinomial Naive Bayes, Linear SVM [cite:35][cite:36][cite:45] |
| Evaluation | scikit-learn metrics [cite:36][cite:37] |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Interface | Jupyter Notebook or simple Streamlit/Flask UI |

## Models Used

For this simple project, the recommended models are:[cite:35][cite:36][cite:45]

| Model | Role in Project | Why Used |
|---|---|---|
| Logistic Regression | Main model | Strong baseline for text classification with TF-IDF features.[cite:28][cite:45] |
| Multinomial Naive Bayes | Baseline model | Fast, simple, and common for text classification.[cite:35] |
| Linear SVM | Optional comparison model | Often performs well on sparse text features such as TF-IDF.[cite:36] |

The best simple choice is to keep **Logistic Regression** as the final main model and use **Naive Bayes** for comparison. This gives a clean academic story: one simple probabilistic baseline and one stronger linear classifier built on the same TF-IDF representation.[cite:35][cite:45][cite:28]

## Suggested Code Structure

A neat folder structure can be:

```text
sentiment-analysis-project/
├── data/
│   └── IMDB Dataset.csv
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── vectorizer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize.py
│   └── main.py
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

Everything can also be kept in one notebook if the subject expects a smaller submission, but separating modules makes the design easier to explain in viva.

## Example Working

A simple example is: the user enters the review “The movie was amazing and I loved the acting.” The system preprocesses this sentence, converts it into TF-IDF features, passes it through the trained Logistic Regression model, and returns the output as **Positive**, while the visualization dashboard can show where this prediction fits relative to the trained dataset distribution and model performance results.[cite:44][cite:45]
