#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import learning_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve)
class ModelInsights:
    """
    Creates visualizations from the data and model:
        model (estimator): The trained machine learning model
        X_train, y_train, X_test, y_test: Training and testing data
        feature_names (List[Str]): All our feature names for plotting
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.predictions = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):  # Check if model supports probability estimates
            self.probas_ = self.model.predict_proba(X_test)
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probas_[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.probas_[:, 1])
        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

    def plot_feature_importance(self):
        try:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(10, 12))  # width, height
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()  # Fixes layout
            plt.show()
        except AttributeError:
            print("No feature_importances_ attribute for this model")

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train, 
                                                                cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        plt.xlabel('Training Sample Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc="best")
        plt.show()

