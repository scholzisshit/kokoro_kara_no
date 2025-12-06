from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def get_model_grids():
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l2']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'model__criterion': ['gini', 'entropy'],
                'model__max_depth': [3, 5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [5, 10, 15],
                'model__min_samples_split': [2, 5],
                'model__max_features': ['sqrt', 'log2']
            }
        },
        'NeuralNetwork': {
            'model': MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
            'params': {
                'model__hidden_layer_sizes': [(32,16), (64,32,16), (50,30)],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__learning_rate_init': [0.001, 0.01]
            }
        }
    }
    return models
