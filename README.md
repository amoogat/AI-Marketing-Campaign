# AI-Marketing-Campaign
Hypertunes a variety of classifiers to optimize call lists for marketing campaigns.

## Recall Metric
I chose the recall metric as the scoring metric since the goal to me was to maximize the true positive rate. This is so that the script can be run on a large unfiltered call list to output a smaller call list that brings in the maximum income. Since we are using recall, we will maximize the actual amount of true positives in our test data. This means when our model predicts the person will donate, they are far more likely to donate than a randomly sampled person.

## Charts
From a high level picture, we can see that the Logistic Regression model performed the best given that we don't want to spend the amount of resources on training a SVM for this process.
### Overall Performance
![Overall](data/metrics.png)
