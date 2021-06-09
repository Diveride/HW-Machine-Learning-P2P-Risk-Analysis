# HW-Machine-Learning-P2P-Risk-Analysis
## Background
In this hw we look to build and evaluate several machine learning models to predict credit risk using data we would typically see from peer-to-peer lending services. "Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans)". This impacts/skews our models training as they will focus on the larger set of loans (good ones). To overcome imbalanced data/classes, we typically employ different techniques to prepare the training data.
In this example we used the imbalanced-learn and Scikit-learn libraries focusing on Resampling and Ensemble Learning techniques. 

## Resampling :
Within the Resampling technique, we compared teh performance of :
1. Simple Logistic Regression before any data manipulation / resampling
2. Oversampling
   1. Used Naive Random Oversampling
   2. SMOTE Oversampling
3. Undersampling (using Cluster Centroids algorithm)
4. Combination (Over and Under) Sampling (using SMOTEEN)

### Conclusion
The Naive Random oversampling SMOTE and Combination model have the same and highest balanced accuracy score = 0.99346.
But the undersampling model has the best recall score with an avg at 0.99 and the lowest False Negative at 113 Vs 124 for the other models. All the models that we resampled (over and under-sampled) produce the same F1 score (also called geometric mean / fowlkes_mallows_score) set at 0.99 (high_risk at 0.91 and low_risk at 1).

We can also add that Resampling the "training data" (whether over or under) resulted in fitting a better model. We can also noticed that the undersample model performed slightly worse than the combined and two oversampled models.

## Ensemble Learning
Within the Ensemble Learning technique, we compared the performance of :
1. Balanced Random Forest Classifier
2. Easy Ensemble Classifier

### Conclusion
The Easy Ensemble Classifier produced the best balanced accuracy score between the two models.
The Balanced Random Forest Classifier produced the best recall score at 0.82 Vs 0.76.
The Balanced Random Forest Classifier produced the best f1 score at 0.89 Vs 0.86.
The top three features are : 'last_pymnt_amnt' (7.3%), 'total_rec_prncp' (7%) and 'total_rec_int' (6.4%).
