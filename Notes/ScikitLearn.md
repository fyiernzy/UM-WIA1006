# Scikit-Learn cookbook

## Recursive Feature Selection (RFE)

rfe.support_ is a boolean array that indicates whether a feature is selected (True) or not selected (False) by the RFE algorithm. A value of True for a feature means that the RFE algorithm has selected that feature as one of the important features for the model, and it will be used in the model training process. A value of False means that the RFE algorithm has not selected that feature, and it will not be used in the model training process.

rfe.ranking_ is an array of integers that indicates the relative importance of each feature, based on the RFE algorithm. The ranking scores are normalized such that the highest ranking score is equal to 1. The lower the ranking score of a feature, the more important it is considered to be by the RFE algorithm. For example, a feature with a ranking score of 1 is considered to be the most important feature by the RFE algorithm, while a feature with a ranking score of 2 is considered to be less important than a feature with a ranking score of 1. If a feature is not selected by the RFE algorithm, its ranking score is set to be equal to the total number of features, which is the highest possible ranking score.

```python
[('area', True, 1),
 ('bedrooms', False, 7),
 ('bathrooms', True, 1),
 ('stories', True, 1),
 ('mainroad', False, 5),
 ('guestroom', False, 6),
 ('basement', False, 4),
 ('hotwaterheating', False, 2),
 ('airconditioning', True, 1),
 ('parking', True, 1),
 ('prefarea', True, 1),
 ('semi-furnished', False, 8),
 ('unfurnished', False, 3)]
```

X_train.columns[rfe.support_] returns a subset of the original dataset's column names, where only the columns corresponding to the features that were selected by the RFE algorithm are included. The boolean array rfe.support_ has a True value for each selected feature, so this indexing expression returns only the names of the selected features.

On the other hand, X_train.columns[~rfe.support_] returns a subset of the original dataset's column names, where only the columns corresponding to the features that were not selected by the RFE algorithm are included. The ~ operator is used to invert the boolean array rfe.support_, so that the indexing expression returns only the names of the non-selected features.

## Histogram of the errors or residuals in the model

```python
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)
plt.xlabel('Errors', fontsize = 18)
```

The code will plot a histogram of the errors or residuals in the model, which is the difference between the actual values of the dependent variable and the predicted values by the model.

The sns.distplot function will create a histogram with a density curve over it, where the x-axis represents the values of the errors and the y-axis represents the frequency or density of those values. The bins parameter specifies the number of bins used to group the error values.

Interpreting the plot, we want the errors to be normally distributed around zero. If the plot shows a skewed or non-uniform distribution, it may indicate that the model does not fit the data well and that there may be other variables or interactions that need to be considered.

