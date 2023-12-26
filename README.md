## Credit Card Fraud Detection

**Author**
Guduru, Thirupathi Reddy (https://github.com/trguduru)

### Executive summary
For the credit card transaction prediction use case, I have used a simulation dataset from *kaggle* and the dataset has 0.5%  of fraud transactions.

I have followed CRISP DM techniques to clean the data for training models. Check data processing and feature engineering for additional details.

The models I have tried are 

* LogisticRegression
* KNN
* DecisionTreeClassifier
* SVM
* GaussianNB
* SGDClassifier

Once trained these models, were evaluated for their performance using f1_score and balanced score of the sklearn metrics library to check for *true positive*.

All these models performed very poorly on *true positive* predictions but *KNN* and *DecisionTreeClassifier* performed relatively better. Check the results section for details.

As stated in the next steps of this README we need to further evaluate additional advanced models and pick a better model for this use case.

### Rationale
The rationale for detecting fraud in a credit card transaction is to eliminate the dollar waste for a financial institution. And also build trust with its customers.
### Research Question
Whether a given credit card transaction is a fraudulent transaction or not.

### Data Sources
The dataset for this use case is here - https://www.kaggle.com/datasets/kartik2112/fraud-detection .
This is simulated data using this simulation tool -  https://github.com/namebrandon/Sparkov_Data_Generation


This is a simulated credit card transaction dataset containing legitimate and fraudulent transactions from the duration of 1st Jan 2019 to 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

### Methodology
I have used the following tools and techniques to classify whether a transaction is fraud or not in machine learning.
#### Data Analysis
This dataset has more than 1M rows and more than 20 features. To understand how each feature's data is and how they are related to each other have done the following analysis using various plots from *matplotlib* , *seaborn* and *plotly*.

Here are some of the interesting plots.

![](images/fraud_by_time.png)

This plot tells that fraud transactions are happening over the night time.

![](images/fraud_distri_cat.png)

This distribution plot shows that high fraud is happening in internet transactions

![](images/fraud_vs_not.png)

This plot shows that there are only 7506 fraud transactions in the dataset which has close to 1.3M records.

#### Data Processing
* Cleaned some of the unnecessary columns like transaction number, merchant name, customer names etc, as these dont add any values in model evaluation.
#### Feature Engineering
* Added a few features like transaction hour from transaction date time, age from dob etc ...
* Applied *JamesSteinEncoder* for encoding categorical data
* Applied *StandardScaler* to scale the dependent features.
* Split the data with a test split size of 30% data.

Here are a few plots in feature engineering.
![](images/heatmap.png)
This heatmap shows that positive and negative correlation between various features. It seems lat, long with merchant lat long has a high positive correlation. Amount and fraud are positive and transaction time and age are in negative correlation.

![](images/feature_imp.png)
This plot shows the important features in predicting whether a transaction is fraud or not.
It shows the amount, transaction time, city etc ... are very important features.
#### Model Evaluation
Evaluated the following models
* LogisticRegression
* KNN
* DecisionTreeClassifier
* SVM
* GaussianNB
* SGDClassifier


### Results
After training all these models using **GridSearchCV** with 5 folds here are the results for various models.

None of these models are accurately predicting *true positive* classes.

Here are some of the plots showing the confusion matrix of various models evaluated.

![](images/log_reg_matrix.png) ![](images/sgd_matrix.png) ![](images/decision_tree_matrix.png)![](images/guassian_matrix.png) ![](images/knn_matrix.png)

These plots show very low *true positive* predictions.

And here is the plot that shows the f1_score and balanced score of these models.

![](images/models_base_perf.png)

### Advanced Techniques

After evaluating the above models and as suggested the next steps documented [here](https://github.com/trguduru/credit_card_fraud_detection_initial#next-steps) trained and evaluated them as stated below.

#### Data Balancing
The target class (is_fraud) is very imbalanced as fraud transactions are only 0.5% of the entire dataset of 1.2M.

Use the following techniques to balance this dataset.
* Random Over Sampling
* BorderlineSMOTE
* SMOTE
* ADASYN

Here are some of the confusion matrixes of these models.
![](images/random_over_sample_conf_martix.png)
![](images/smote_cfm.png) ![](images/borderline_smote_cfm.png)

All these oversampling models accurately predict *true positive* but these models seem to overfit as data duplication with oversampling techniques and may not perform very well on never seen data.

#### Ensemble Models
Ensemble techniques and models might improve the accuracy of the models so we can try the following

* BaggingClassifier
* AdaBoostClassifier
* BalancedRandomForestClassifier
* RUSBoostingClassifier
* RandomForestClassifier

Evaluated the above-stated models and here are a few confusion matrixes that talk about how these perform in predicting *true positives*.

![](images/bagging_classifier_cfm.png) ![](images/adaboosting_cfm.png)![](images/rus_cfm.png)![](images/balanced_rfc_cfm.png)

These are some of the plots of the above models and the model **BalancedRandomForestClassifier** predicts more than 95% of *true positive* data.


#### Neural Networks
Applying deep neural network *Dense* models along with cross-validation could also improve the accuracy.

Evaluated deep learning models with and without hyperparameter tuning and here are some of the plots and results of them.

![](images/neral_net_basic_epochs_perf.png)

This plot shows the performance of a basic deep-learning model with epochs in terms of loss and accuracy. If we continue training the model with more and more epochs it seems the performance is increasing but not accurately predicting *true positives*.

Here is the network architecture of this model
![](images/base_deep_learning_graph.png)

![](images/base_deep_learning_cfm.png)

The basic deep learning model with 2 hidden layers predicted *true positives* close to 70% only.

##### Hyperparameter Tuning

Tuned the deep learning model with **RandomizedSearchCV** and here is the neural network architecture diagram.

![](images/optimized_deep_learning_graph.png)

Added a few regularization layers( Dropouts with 0.2 and 0.1) and also added a few more hidden layers.

![](images/optimized_deep_learning_cfm.png)

The above confusion matrix shows the hyperparameter tuned with **RandomizedSearchCV** and its prediction is even lower than the basic deep learning model.



### Final Results

Here is the plot which shows all the models trained on this dataset.

![](images/final_perf.png)

It shows that ensemble and over-sampling models are performing better than other models.

The goal of this project is to predict a fraud transaction accurately as this will save money for the financial institutions.


#### Best Model
After evaluating all these models the **BalancedRandomForestClassifier** is doing way better than others.

Here are the results that are run on a test dataset which this model has never seen.

![](images/final_cfm.png)

The above confusion matrix shows that it predicts more than 95% accuracy in fraud transactions.

### Outline of project

#### Project Organization
Here are various folders in this project containing the files of this project.
* data, it has train and test datasets
* images, contains various images of the data analysis and model performance plots etc ...
* jupyter notebook
* README
* git helper files
* vscode workspace

#### Deliverables
- [Jupyter Notebook](credit_card_fraud_detection.ipynb)

#### Library Requirements
To run this project you need to install the following libraries
* default anaconda environment
* intel sklearn extensions
* category_encoders
* keras_visualizer
* imblearn
* sklearn_evaluation

### Contact and Further Information
* Guduru, Thirupathi Reddy (https://github.com/trguduru)
