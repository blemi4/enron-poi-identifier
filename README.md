# Enron Person of Interest Classifier

The goal of this project is to create a machine learning model to predict whether or not a person in the Enron dataset is a person of interest (POI) to the US Justice Department from the Enron corporate fraud case. Read more about the Enron scandal here: https://en.wikipedia.org/wiki/Enron_scandal

In this README I will give an overview of my process and many of the important aspects of the project.  For a more indepth description of my entire process and the code associated with the project visit: https://github.com/blemi4/enron-poi-identifier/blob/master/poi_identifier_enron.ipynb

### Summary of the Dataset
The goal of this project is to use a machine learning model to predict whether or not a person in the Enron dataset is a person of interest (POI).  Machine learning is great at making predictions on data where there is a pattern.  The algorithms can identify patterns among many inputs/features that are very difficult for humans to identify.

The dataset consists of 24 features including several different financial features (such as exercised stock options) and several email features (such as emails from the person to a POI).  There are 145 data points total, 18 of which represent POI’s.  Several of the data points have missing values for some of their fields – I’ve replaced these with zeros.  I discovered an outlier in the data – it is the ‘TOTAL’ of all other data points for each feature.  I discovered this by visualizing expenses vs. salary:

# Insert Picture

I wrote a short line of code to remove ‘TOTAL’  and reran the plot:

# Insert Picture

### Feature Selection
The following features were used to create a prediction algorithm to classify POI’s (note: this is almost all of the features plus the three highlighted features which I created): ['salary','expenses', 'pct_to_poi', 'pct_from_poi', 'pct_shared_poi', 'exercised_stock_options', 'deferral_payments', 'deferred_income', 'director_fees', 'loan_advances', 'long_term_incentive', 'bonus','other', 'restricted_stock', 'restricted_stock_deferred', 'total_payments', 'total_stock_value'].  My final algorithm is a hybrid of the other three algorithms.  For each of the three, I used scikit-learn’s SelectPercentile function and GridSearchCV function to test which percentile of features to use for each of the algorithms.  Only one of the three algorithms didn’t select 100 percent of the features.  The K Nearest Neighbor model selected 30% of the features:
•	Features selected: ['pct_shared_poi', 'total_payments', 'long_term_incentive', 'expenses'] 
•	Accompanying scores:  [24.815, 24.183, 20.792, 18.290]
The Extreme Gradient Boosting model selected 85% of features:
•	Features selected: ['pct_shared_poi', 'total_payments', 'long_term_incentive', 'expenses', 'deferral_payments', 'loan_advances', 'other', 'pct_from_poi', 'restricted_stock_deferred', 'director_fees', 'salary', 'bonus', 'pct_to_poi', 'deferred_income']
•	Accompanying scores:  [24.815, 24.183, 20.792, 18.290, 16.410, 11.458, 9.922, 9.213, 9.101, 8.772, 7.184, 6.094, 4.187, 3.128]

I created three new features: percentage of emails to a POI, percentage of emails from a POI, and percentage of emails shared with a POI.  These scaled features were created from the absolute number of emails to, from, and shared with a POI divided by either the total number of emails from or to the individual.  These features are preferable to the non-scaled versions because their relationship to each other is more telling than the raw numbers.  I also used feature scaling on all three of my algorithms.  Usually, feature scaling would only be necessary for KNeighborsClassifier, because it is based on the Euclidean distance between data points.  However, I used Principal Component Analysis (PCA) in conjunction with feature selection.  PCA essentially compresses the information from several features into a much smaller amount of features.  It does this by comparing the variance of different features, therefore having scaled features to input into PCA is very important.

A note on my created features:  The features I created are based on the number of emails the person has from, to, or shared receipt with POI’s.  This assumes that we already know who all the POI’s are.  However, if we were truly trying to predict a new data point, we wouldn’t know whether or not that person is POI, therefore they could not be accounted for in the email data.  This presents a potential leakage problem with training data and testing data splits.  For the purposes of this problem, we will assume that the email features represent the relationship with know POI’s, not all POI’s.  This is an imperfect assumption, but practical all things considered. 

### Algorithm Testing and Selection
I tested 3 different classification algorithms: Gaussian Naïve Bayes (GaussianNB), Extreme Gradient Boosting (XGBoost) and K Nearest Neighbors (KNeighborsClassifier).  I then used scikit-learn’s VotingClassifer function to create a hybrid of all three, which I ended using.  Here are the results:
Classifier	Precision	Recall
GaussianNB	.2085	.7950
XGBoost	.4732	.3260
KNeighborsClassifier	.3604	.3130
Hybrid Classifier (final)	.4099	.4560

Among the first three classifiers there are pretty large tradeoffs between precision and recall, (a description of these two evaluation metrics will be provided in Part 6) however the hybrid achieves a solid score on both metrics.

### Parameter Tuning
Tuning parameters of the algorithm essentially means to optimize the parameters of the algorithm your using in order for it to perform best on a test set.  If the tuning is not done well, the algorithm could potentially overfit/underfit.  

I tuned XGBoost and KNeighborsClassifier for use in the final Hybrid Classifier (GaussianNB does not have any parameters to tune).  To do so, I used GridSearchCV – a function that allows the user to pass in a number of different variables for each parameter the algorithm takes, and provides a score on a user-defined metric using a user defined cross validation technique (validation will be discussed further in Part 5).  I passed in several variables for each parameter, set GridSearchCV to optimize recall (evaluation metrics will be discussed further in Part 6), and set the cross validation to use StratifiedShuffleSplit with 1,000 iterations (again this will be discussed more in Part 5).

### Validation
Validation essentially means to test your algorithm on data it was not trained on to verify that the results are “valid” in the sense that the model has the correct fit.  It would be a classic mistake to train and validate the algorithm on the same data.  This will likely lead to over-fitting the algorithm and poor performance on data points outside of the initial dataset.

I decided to use scikit-learn’s StratifiedShuffleSplit function with 1,000 folds to validate my classifier.  This cross validation technique is especially preferable for this problem because the sample size is relatively small and classification labels are lopsided (many Non-POI’s, few POI’s).  This function essentially takes 1,000 random stratified samples of training data and testing data from the dataset.  I was then able to take the average of several different evaluation metrics from the cross-validated data.

### Evaluation
Evaluation metrics I care most about for this problem are recall and precision.  Recall is the percentage of POI’s that we correctly identified out of all of the POI’s in the dataset.  A recall of 1.0 would mean that all POI’s in the dataset were classified as such by our model.  The recall of my Hybrid Classifier is 0.475.  Precision is the percentage of people we classify as POI’s that are actually POI’s.  A precision of 1.0 would mean that all of the people the model classifies as POI’s are actually POI’s.  The precision of my Hybrid Classifier is 0.407.

In my opinion, recall is slightly more important than precision for this problem.  Investigators would likely prefer to catch as many POI’s in their net as possible, even if that means some Non-POI’s are included.  At this point a human investigation by the Justice Department could sort out who is actually a POI or not.  Of course, precision is still important because each person that has to be investigated further by humans will cost time and money, as well as potentially causing emotional harm to the falsely suspected.












