import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

def plot_model_comparison(results_dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    data = []
    for model_name, metrics in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })
    df_metrics = pd.DataFrame(data).melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_metrics, x='Metric', y='Score', hue='Model', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    return fig

def generate_wordcloud(text_data, title="Word Cloud"):
    fig, ax = plt.subplots(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(" ".join(text_data))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    return fig

def plot_class_distribution(df, label_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=label_col, ax=ax, palette='Set2')
    ax.set_title('Class Distribution')
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    return fig
