# Sentiment Analysis Dashboard

A machine learning project for classifying text reviews as positive or negative using Natural Language Processing, TF-IDF feature extraction, and traditional classification models. The project includes a Streamlit dashboard for real-time sentiment prediction, dataset exploration, model evaluation, and visual comparison of classifier performance.

## Features

- Real-time sentiment prediction for custom text input
- Text preprocessing with lowercasing, HTML removal, punctuation removal, stopword filtering, and lemmatization
- TF-IDF based feature extraction with unigram and bigram support
- Logistic Regression model for primary sentiment classification
- Multinomial Naive Bayes model for baseline comparison
- Model evaluation using accuracy, precision, recall, F1-score, classification reports, and confusion matrices
- Interactive Streamlit dashboard with prediction, data exploration, and model evaluation pages
- Word cloud visualization for positive and negative reviews
- Class distribution and model comparison charts

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python |
| Data Handling | pandas |
| NLP | NLTK, regular expressions |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn, WordCloud, Altair |
| Web App | Streamlit |
| Model Storage | joblib |

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── IMDB Dataset.csv
├── models/
│   ├── sentiment_model_lr.pkl
│   ├── sentiment_model_nb.pkl
│   └── test_data.pkl
└── src/
    ├── evaluate.py
    ├── preprocessing.py
    ├── train.py
    └── visualize.py
```

> Note: The `models/` directory is generated after training and may not exist in a fresh clone.

## Dataset

This project is designed to work with the IMDb movie review dataset. The expected file path is:

```text
data/IMDB Dataset.csv
```

The dataset should contain review text and sentiment labels. By default, the code looks for these columns:

- `review`
- `sentiment`

If those exact names are not found, the project falls back to using the first column as text and the second column as the label.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the dataset at:

```text
data/IMDB Dataset.csv
```

## Training the Models

Run the training script from the project root:

```bash
python src/train.py
```

This will:

- Load the IMDb dataset
- Clean and preprocess the review text
- Split the data into training and testing sets
- Train Logistic Regression and Naive Bayes models
- Save trained models and test data inside the `models/` directory

Generated files:

```text
models/sentiment_model_lr.pkl
models/sentiment_model_nb.pkl
models/test_data.pkl
```

## Evaluating the Models

Run:

```bash
python src/evaluate.py
```

The evaluation script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

## Running the Streamlit App

After training the models, start the dashboard:

```bash
streamlit run app.py
```

The dashboard includes three sections:

- **Prediction**: Enter custom text and get a live sentiment prediction.
- **Data Exploration**: View dataset samples, class distribution, and word clouds.
- **Model Evaluation**: Compare model performance and inspect confusion matrices.

## Machine Learning Workflow

```text
Raw Text Reviews
        ↓
Text Cleaning and Lemmatization
        ↓
TF-IDF Feature Extraction
        ↓
Model Training
        ↓
Sentiment Prediction
        ↓
Evaluation and Visualization
```

## Example Prediction

Input:

```text
The movie was absolutely fantastic and I loved every minute of it.
```

Expected output:

```text
Positive
```

## Notes

- The first run may download required NLTK resources such as stopwords and WordNet.
- If model files are missing, run `python src/train.py` before launching the dashboard.
- Large datasets and generated model files can be excluded from GitHub using `.gitignore` if needed.

## Future Improvements

- Add support for neutral sentiment classification
- Include deep learning models such as LSTM, GRU, or BERT
- Add batch prediction from uploaded CSV files
- Save evaluation charts as image files
- Add automated tests for preprocessing and prediction logic

## License

This project is available for educational and portfolio use. Add a license file before using it in production or distributing it publicly.
