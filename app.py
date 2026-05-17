import os
import joblib
import pandas as pd
import streamlit as st
from st_keyup import st_keyup

# Add src to path so we can import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import clean_text
from evaluate import evaluate_models
from visualize import plot_confusion_matrix, plot_model_comparison, generate_wordcloud, plot_class_distribution

# Setup Streamlit page
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide", page_icon="📊")

st.title("Sentiment Analysis Dashboard 📊")
st.markdown("Analyze sentiment of text reviews using Natural Language Processing and Machine Learning.")

# Load models and data
@st.cache_resource
def load_resources():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    data_path = os.path.join(os.path.dirname(__file__), 'data/IMDB Dataset.csv')
    
    lr_path = os.path.join(models_dir, 'sentiment_model_lr.pkl')
    nb_path = os.path.join(models_dir, 'sentiment_model_nb.pkl')
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    
    resources = {
        'lr_model': joblib.load(lr_path) if os.path.exists(lr_path) else None,
        'nb_model': joblib.load(nb_path) if os.path.exists(nb_path) else None,
        'test_data': joblib.load(test_data_path) if os.path.exists(test_data_path) else None,
        'dataset_exists': os.path.exists(data_path),
        'data_path': data_path
    }
    return resources

resources = load_resources()

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Data Exploration", "Model Evaluation"])

if page == "Prediction":
    st.header("Predict Sentiment of Custom Text")
    
    if not resources['lr_model']:
        st.warning("Model not found. Please run `train.py` first.")
    else:
        user_input = st_keyup("Enter a movie review or any text:", placeholder="The movie was absolutely fantastic! I loved every minute of it.", debounce=250)
        
        if user_input.strip() != "":
            with st.spinner("Analyzing..."):
                # Preprocess
                cleaned = clean_text(user_input)
                
                # Predict
                model = resources['lr_model']
                prediction = model.predict([cleaned])[0]
                probabilities = model.predict_proba([cleaned])[0]
                
                # model.classes_ might be ['negative', 'positive'] or [0, 1]
                classes = list(model.classes_)
                pred_idx = classes.index(prediction)
                
                sentiment = "Positive" if prediction in [1, "positive", "Positive"] else "Negative"
                confidence = probabilities[pred_idx]
                
                st.subheader("Result")
                if sentiment == "Positive":
                    st.success(f"**Sentiment:** {sentiment} 😊 (Confidence: {confidence:.1%})")
                else:
                    st.error(f"**Sentiment:** {sentiment} 😞 (Confidence: {confidence:.1%})")
                
                st.markdown("### Under the Hood")
                st.text(f"Cleaned Tokens: {cleaned}")
                
                # Extract Word Weights
                try:
                    lr = model.named_steps['clf']
                    tfidf = model.named_steps['tfidf']
                    vocab = tfidf.vocabulary_
                    
                    word_weights = []
                    for word in set(cleaned.split()):
                        if word in vocab:
                            weight = lr.coef_[0][vocab[word]]
                            word_weights.append({"Word": word, "Weight": weight})
                    
                    if word_weights:
                        st.markdown("#### Word Contributions (Weights)")
                        st.markdown("Positive weights push the prediction towards **Positive**, negative weights push towards **Negative**.")
                        weight_df = pd.DataFrame(word_weights).sort_values(by="Weight", ascending=False)
                        
                        import altair as alt
                        chart = alt.Chart(weight_df).mark_bar().encode(
                            x=alt.X('Weight:Q', title='Model Weight'),
                            y=alt.Y('Word:N', sort='-x', title='Word'),
                            color=alt.condition(
                                alt.datum.Weight > 0,
                                alt.value("green"),
                                alt.value("red")
                            )
                        ).properties(height=max(200, len(weight_df) * 30))
                        st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    pass
        else:
            st.info("Start typing above to see real-time analysis!")

elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if not resources['dataset_exists']:
        st.warning(f"Dataset not found at `data/IMDB Dataset.csv`.")
    else:
        st.info("Loading dataset...")
        df = pd.read_csv(resources['data_path'])
        
        text_col = 'review' if 'review' in df.columns else df.columns[0]
        label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]
        
        st.subheader("Dataset Sample")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            fig_dist = plot_class_distribution(df, label_col)
            st.pyplot(fig_dist)
            
        with col2:
            st.subheader("Word Clouds")
            if st.button("Generate Word Clouds (Takes a few seconds)"):
                with st.spinner("Generating Word Clouds..."):
                    pos_text = df[df[label_col].isin(['positive', 1])][text_col].sample(min(1000, len(df)), random_state=42).tolist()
                    neg_text = df[df[label_col].isin(['negative', 0])][text_col].sample(min(1000, len(df)), random_state=42).tolist()
                    
                    st.markdown("**Positive Reviews**")
                    fig_pos = generate_wordcloud(pos_text, "Positive Words")
                    st.pyplot(fig_pos)
                    
                    st.markdown("**Negative Reviews**")
                    fig_neg = generate_wordcloud(neg_text, "Negative Words")
                    st.pyplot(fig_neg)

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    if not resources['test_data'] or not resources['lr_model']:
        st.warning("Models or test data not found. Please run `train.py` first.")
    else:
        with st.spinner("Evaluating models..."):
            # We can use our evaluate.py logic
            # To avoid reloading from disk in evaluate.py, we pass the loaded objects
            # Actually, evaluate.py reads from disk. Let's just run it
            results_dict, y_test = evaluate_models()
            
            if results_dict:
                st.subheader("Model Comparison")
                fig_comp = plot_model_comparison(results_dict)
                st.pyplot(fig_comp)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Logistic Regression Confusion Matrix")
                    fig_cm_lr = plot_confusion_matrix(y_test, results_dict["Logistic Regression"]["y_pred"], "Logistic Regression")
                    st.pyplot(fig_cm_lr)
                    
                with col2:
                    st.subheader("Naive Bayes Confusion Matrix")
                    fig_cm_nb = plot_confusion_matrix(y_test, results_dict["Naive Bayes"]["y_pred"], "Naive Bayes")
                    st.pyplot(fig_cm_nb)
                    
                st.subheader("Detailed Metrics")
                metrics_data = []
                for model_name, metrics in results_dict.items():
                    metrics_data.append({
                        "Model": model_name,
                        "Accuracy": f"{metrics['accuracy']:.4f}",
                        "Precision": f"{metrics['precision']:.4f}",
                        "Recall": f"{metrics['recall']:.4f}",
                        "F1-Score": f"{metrics['f1']:.4f}"
                    })
                st.table(pd.DataFrame(metrics_data))
                
                st.subheader("Classification Reports")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Logistic Regression**")
                    st.text(results_dict.get("Logistic Regression", {}).get("report", "N/A"))
                with col4:
                    st.markdown("**Naive Bayes**")
                    st.text(results_dict.get("Naive Bayes", {}).get("report", "N/A"))
