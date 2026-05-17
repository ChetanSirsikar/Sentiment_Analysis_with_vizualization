import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from preprocessing import clean_text

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/IMDB Dataset.csv')
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print("Dataset not found. Please ensure it's placed in 'data/IMDB Dataset.csv'")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    
    # We expect columns like 'review' and 'sentiment'. 
    # If they are named differently, we map them.
    text_col = 'review' if 'review' in df.columns else df.columns[0]
    label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

    # Convert sentiment labels to numeric (0 and 1)
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({'positive': 1, 'negative': 0})
        
    print("Sample data:")
    print(df.head(2))
    
    print("\nCleaning text data (this may take a while)...")
    # Clean only a subset for faster testing if needed, but we'll do all
    df['cleaned_text'] = df[text_col].apply(clean_text)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df[label_col], test_size=0.2, random_state=42
    )

    print("\nBuilding and training Logistic Regression pipeline...")
    # Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    print("Building and training Naive Bayes pipeline...")
    # Naive Bayes Pipeline (for comparison)
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)

    # Save models
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    lr_model_path = os.path.join(os.path.dirname(__file__), '../models/sentiment_model_lr.pkl')
    nb_model_path = os.path.join(os.path.dirname(__file__), '../models/sentiment_model_nb.pkl')
    test_data_path = os.path.join(os.path.dirname(__file__), '../models/test_data.pkl')
    
    joblib.dump(lr_pipeline, lr_model_path)
    joblib.dump(nb_pipeline, nb_model_path)
    joblib.dump((X_test, y_test), test_data_path)
    
    print(f"Models saved to:\n- {lr_model_path}\n- {nb_model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()
