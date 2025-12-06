import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             RocCurveDisplay)
from sklearn.pipeline import Pipeline
from src.data_prep import preprocess_pipeline
from src.models import get_model_grids

def train_all_models(X_train, X_test, y_train, y_test):
    preprocessor, _, _ = preprocess_pipeline()
    models = get_model_grids()
    results = {}
    
    for name, config in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        grid = GridSearchCV(pipe, config['params'], cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        # Test predictions
        y_pred = grid.best_estimator_.predict(X_test)
        y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Save best model
        joblib.dump(grid.best_estimator_, f'../models/{name.lower()}_best.pkl')
    
    return results

def plot_results(results):
    # Performance table
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1'] for r in results.values()],
        'ROC-AUC': [r['roc_auc'] for r in results.values()]
    }).round(3)
    print(metrics_df)
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        RocCurveDisplay.from_estimator(r['best_model'], X_test, y_test, ax=ax, name=name)
    plt.title('ROC Curves Comparison')
    plt.savefig('../reports/figures/roc_curves.png', dpi=300)
    
    # Feature importance (Random Forest)
    rf_model = results['RandomForest']['best_model']
    importances = rf_model.named_steps['model'].feature_importances_
    feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=True).tail(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='importance', y='feature')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.savefig('../reports/figures/feature_importance.png', dpi=300)
