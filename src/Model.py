#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

class Model:
    """
    Responsible for training and tuning machine learning models:
        model (estimator): The machine learning model to be trained and tuned
        tuning_params (dict): Hyperparameters for the model to tune
        use_random_search (bool): Flag to decide between using RandomizedSearchCV or GridSearchCV
    """
    def __init__(self, model, tuning_params, use_random_search=False):
        # Initializes Model obj with a specific machine learning model & tuning params
        self.model = model
        self.tuning_params = tuning_params # Dict with ranges for params
        self.use_random_search = use_random_search
        self.training_time = 0
        
    def fit_model(self, X_train, y_train):
        # Fits the model, selects best hyperparameters 
        start_time = time.time()  # Start timing
        if self.use_random_search:
            # Use RandomizedSearchCV for larger parameter spaces
            search = RandomizedSearchCV(self.model, self.tuning_params, scoring='recall',random_state=42,
                                       n_jobs=-1,n_iter=3,cv=3)
        else:
            # Use GridSearchCV for exhaustive search
            search = GridSearchCV(self.model, self.tuning_params, scoring='recall',cv=5)
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.training_time = time.time() - start_time  # Total training time
        print(f"Best Parameters: {search.best_params_}")
        # Return hypertuned model
        return self.model

    def evaluate(self, X_test, y_test):
        #Evaluates the trained model on the test set using various metrics
        predictions = self.model.predict(X_test)
        results = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average='macro', zero_division=0),
            "Recall": recall_score(y_test, predictions, average='macro', zero_division=0),
            "F1 Score": f1_score(y_test, predictions, average='macro'),
            "Training Time": self.training_time
        }
        if hasattr(self.model, "predict_proba"):  # Check if model supports probability estimates
            probabilities = self.model.predict_proba(X_test)[:, 1]
            results['ROC AUC'] = roc_auc_score(y_test, probabilities)
        #Returns dictionary containing accuracy, precision, recall, and F1 score
        return results

