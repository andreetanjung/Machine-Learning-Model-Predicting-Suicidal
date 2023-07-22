# Machine-Learning-Model-Predicting-Suicidal
### Project Overview
Supervised Learning with Best Model Predicting Suicidal Based on Depression and Anxiety

In this notebook we will attempt to predict whether the individuals on dataset would likely to commit suicide or not based on their anxiety and depression using a supervised learning model. The dataset is about individuals and their mental health, with information such as their demographic, depression, anxiety and stress scale and whether individuals are diagnosed with depression or anxiety by medical professional. The objective is to develop classification models that can accurately distinguish the target prediction with the highest possible score. 
### Data 
The data was obtained from Kaggle which I forgot to copy the link, but you can download and see it as `anxiety and depression.csv`
### Methodology
- Data preprocessing: make sure the data has no duplicate and missing value
- Exploratory data: understand more about the data with visualization
- Feature engineering and selection: using pipeline to handle missing value, outlier, and encoding. For the feature selection we use phik matrix, recursive feature elimination, and mutual info classification to compare and combine.
- Modeling: comparing logistic regression, SVM, decision tree, random forest, KNN, Naive Baiyes and Adaboost model to find the best model
### Result and Evaluation
| Model | Result |
| --- | --- |
|Evaluating Logistic Regression using cross-validation|Mean precision score: 0.000 +/- 0.000|
|Evaluating SVM using cross-validation|Mean precision score: 0.000 +/- 0.000|
|Evaluating Decision Tree using cross-validation|Mean precision score: 0.607 +/- 0.277|
|Evaluating Random Forest using cross-validation|Mean precision score: 0.619 +/- 0.452|
|Evaluating KNN using cross-validation|Mean precision score: 0.048 +/- 0.117|
|Evaluating Naive Bayes using cross-validation|Mean precision score: 0.363 +/- 0.119|
|Evaluating AdaBoost using cross-validation|Mean precision score: 0.524 +/- 0.382|

Notably, all models exhibited overfitting as the training scores were significantly higher than the corresponding test scores

![image](https://github.com/andreetanjung/Machine-Learning-Model-Predicting-Suicidal/assets/123824152/ec22c79a-8340-4f25-a19f-88d541d90ccc)

Unfortunately, hyperparameter tuning did not result in any improvement in the model performance. Comparing the results before and after tuning, the models in fact has better precision and f1 score beforehand, and both are still overfit. Based on the model selection, while the models have good accuracy in classifying individuals in class 0, they are not effective in identifying individuals in class 1, which is our main objective in this dataset. Therefore, we cannot recommend deploying the model in its current state.
### Future References
Future work can explore different features that may provide more predictive power, and also collect more real data on suicidal individuals as the SMOTENC technique may not be sufficient to address the class imbalance issue. Additionally, alternative models can also be explored to improve the classification performance.
### Directory Structure and Brief Description of Files
`h8dsft_P1M2_Andrian_Tanjung.ipynb` is a Jupyter notebook that contains the complete project code, including data pre-processing, feature engineering with model, and evaluation.
Model deployment: https://huggingface.co/spaces/andreetanjung/Milestone2
