#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearnex import patch_sklearn 
from src.Model import Model
from src.ModelInsights import ModelInsights
from src.DataHandler import DataHandler


# In[10]:


def plot_accuracy(results_df):
    # Setting the style
    plt.style.use('seaborn-darkgrid')
    # Plotting Train and Test Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'], ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Metrics')
    ax.set_xlabel('Model')
    ax.set_ylim([0, 1])  # Assuming accuracy is between 0 and 1
    plt.xticks(rotation=45)
    plt.legend(title='Metric Type')
    plt.show()
    


# In[11]:


def plot_train_time(results_df):
    # Plotting Training Time
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', x='Model', y='Training Time', color='teal', ax=ax)
    ax.set_title('Model Comparison - Training Time')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(['Training Time'])
    plt.show()
    


# In[12]:


def main():
    patch_sklearn()
    # Define models and hyperparameter grids
    model_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
            'use_random_search': False  # GridSearchCV is suitable for smaller parameter space
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 9]},
            'use_random_search': False  # Exhaustive search is manageable for KNN
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10],
                'max_features': ['auto', 'sqrt', 'log2', None]
            },
            'use_random_search': False  
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {'C': [1, 7], 'kernel': ['rbf', 'linear']},
            'use_random_search': True  # RandomizedSearchCV for larger parameter space efficiency
        }
    }
    
    # Prepare and split data using DataHandler Class
    data_handler = DataHandler('data/marketing.csv', ['duration'], 'y_yes')
    features, labels = data_handler.load_and_prepare()
    X_train, X_test, y_train, y_test = data_handler.split_data()
    feature_names = features.columns.tolist()
    
    # Train and evaluate models, compile results for comparison
    results = []
    important_features_dict = {}
    for name, config in model_config.items():
        trainer = Model(config['model'], config['params'], config['use_random_search'])
        trained_model = trainer.fit_model(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        metrics['Model'] = name  # Adds model name to the dictionary
        results.append(metrics)
                       
        # Plots for the individual model
        insights = ModelInsights(trained_model, X_train, y_train, X_test, y_test, feature_names)
        insights.plot_confusion_matrix()
        if hasattr(trained_model, "predict_proba"):  # Check if model supports probability estimates
            insights.plot_roc_curve()
            insights.plot_precision_recall_curve()
        if hasattr(trained_model, 'feature_importances_') or hasattr(trained_model, 'coef_'):
            insights.plot_feature_importance()
        insights.plot_learning_curve()
    # Output results to review model performances
    results_df = pd.DataFrame(results)
    print(results_df)
    plot_accuracy(results_df)
    plot_train_time(results_df)
    


# In[13]:


if __name__ == "__main__":
    main()

