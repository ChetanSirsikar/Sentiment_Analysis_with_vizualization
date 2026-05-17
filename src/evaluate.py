import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_models():
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    lr_model_path = os.path.join(models_dir, 'sentiment_model_lr.pkl')
    nb_model_path = os.path.join(models_dir, 'sentiment_model_nb.pkl')
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    
    if not os.path.exists(test_data_path) or not os.path.exists(lr_model_path):
        print("Models or test data not found. Please run train.py first.")
        return None
        
    print("Loading test data and models...")
    X_test, y_test = joblib.load(test_data_path)
    lr_model = joblib.load(lr_model_path)
    nb_model = joblib.load(nb_model_path)
    
    results = {}
    
    for name, model in [("Logistic Regression", lr_model), ("Naive Bayes", nb_model)]:
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        
        pos_label = 1 if 1 in y_test.values else 'positive'
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=pos_label)
        rec = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
            'report': report
        }
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        
    return results, y_test

if __name__ == "__main__":
    evaluate_models()
