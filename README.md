# ml_ANZ-Customer-Transaction-Prediction
Predict the customer's annual revenue from ANZ's read transaction data
## ANZ Customer Transaction Prediction
Review the current status of Internet Banking in Australia. about Internet Banking and examining the current status with respect to banking on the internet, the services they provide, the difficulties faced by customers, and the corrective actions required by the banks, ending with the observation that Australian banks are lagging behind their US, European counterparts and Japan in providing banking services on the Internet and calls for serious efforts by banks, computer specialists, academics and others to popularize this upcoming area if banking does not look to be medieval in Australia by comparison to World standards[1]. The progressive deregulation of the Australian financial system, announced by the Committee of Inquiry at Campbell (1981), has created enormous pressure for structural adjustment in the Australian banking industry. ANZ's recent move to acquire Grindlays, a UK-based international bank, was the first significant attempt at a thorough internationalization by an Australian bank. The takeover is analyzed from the bank internationalization model developed by Fujita and Ishigaki. Of particular interest to ANZ is the foreign bank representation that Grindlays has. This not only complements ANZ's existing international representation, but also allows Australian buyers to overcome reciprocal regulatory barriers that prevent Australian banks from operating in many offshore locations. By acquiring Grindlays Bank, ANZ gained an internationalization advantage over its Australian rivals. Its full potential will only be seen after the new activity package brought by Grindlays Bank has been fully integrated and rationalized. ANZ has gained an internationalization advantage over its Australian rivals. Its full potential will only be seen after the new activity package brought by Grindlays Bank has been fully integrated and rationalized[2]. The task datasheet is based on a synthesized transaction dataset, containing 3 months of transactions for 100 hypothetical customers, containing purchases, recurring transactions & payroll transactions. The data set is designed to simulate the realistic transaction behavior observed in ANZ read transaction data. The main targets in this task are: Generate target variables for the problem & create a model that can predict the customer's annual revenue.

## Business Understanding
### Problem Statements
* How to predict customer annual revenue from ANZ read transaction data?
* Which evaluation model is the most accurate for predicting transaction data from ANZ?

### Goals
* Generate target variables for problems & create models that can predict comprehensive customer annual revenue with EDA techniques.
* Makes some algorithm model recommendations, then tries to find features that will improve the model using the most accurate methodology

### Solution statements
* Using Exploratory data analysis or often abbreviated as EDA for the initial investigation process on the data to analyze characteristics, find patterns, anomalies, and check assumptions in the data.
* Develop machine learning models with three algorithms. Then, evaluate the performance of each algorithm and determine which algorithm gives the best predictive results. The three algorithms that we will use include: *K-Nearest Neighbor*, *Random Forest*, *Boosting Algorithm*

## Data Understanding
Synthesized transaction data set containing 3 months of transactions for 100 hypothetical customers. This data set is designed to simulate the realistic transaction behavior observed in ANZ's real transaction data of purchases, recurring transactions, and payroll transactions.
ANZ data source https://www.kaggle.com/datasets/ashraf1997/anz-synthesised-transaction-dataset

The variables in the ANZ dataset are as follows:
* merchant_code: identify the merchant code
* merchant_id: merchant identifier (most common 106e1272-44ab-4dcb-a438-dd98e0071e51)
* merchant_latitude: merchant latitude (most common -37.82)
* merchant_longitude: merchant's longitude (most common 151.21)
* merchant_state: where it is located (NSW is the most common)
* merchant_suburb: the specific suburb where it is located (Melbourne most common)
* amount: the number of transactions
* movement: type of transaction, credit or debit (debit related to expenses, credit related to payment of salaries)
* status: transaction status, maybe some transactions have not been approved (posted)
* transaction_id: unique identifier for each transaction made
* txn_description: payment category
* card_present_flag: possibly indicating whether the payment was made virtually or physically
* date: data on when the transaction was made, the busiest is 9/28/2018
* extraction: another date, possibly the correct time is included (but needs to be extracted), not just the date
* bpay_biller_code: bpay usually has a unique value
* long_lat: the location associated with the transaction
* account: account number, dataset containing a total of 100 unique bank accounts & their transactions
* customer_id: 100 unique customer identifiers
* first_name: The most common first name associated with the transaction is Michael
* age: most transactions are done by the age of 26 years
* balance: the balance before the transaction occurred, or maybe after, the former is more reasonable
* gender: the sex of the transaction

Performing several steps needed to understand the data, EDA Handles Missing Value, Univariate Analysis, Exploratory Data Analysis - Multivariate Analysis
* EDA Handles Missing Value

![pi](https://user-images.githubusercontent.com/123156703/215402738-11befd36-54df-49d0-a0c8-2cb2ee2b4859.png)

Figure 1. NAN values ​​in the datasheet

From the results of pd.read or the results of the describe() function, there are values ​​that need to be deleted in the dataseeet merchant_kode, bpay_biller_code, and card_present_flag using the drop function. then use the Outliers function.

![outliers](https://user-images.githubusercontent.com/123156703/215409314-e032a12c-f095-4492-bbf4-7db3a85b1078.png)

Figure 2. Boxplot with Value Balance


![outliers2](https://user-images.githubusercontent.com/123156703/215409317-178a5204-ef31-444c-a4f4-0fbbb31aca34.png)

Figure 3. Boxplot with Age Value


![outliers3](https://user-images.githubusercontent.com/123156703/215409318-fa6a56c4-f51a-43cd-9b7b-08b053350b24.png)

Figure 4. Boxplot with Value Amount



Outliers are samples whose values ​​are very far from the general scope of the main data, are the results of observations that occur very rarely and are different from other observed data. in this experiment it turns out that there are outliers in the ANZ data, then, the action to overcome the outliers is with the equation function:

 |Upper limit = Q3 + 1.5 * IQR |
    | ------ |


   | Lower limit = Q1 - 1.5 * IQR |
    | ------ |

* EDA Univariate Analysis
Furthermore, the process of data analysis using the Univariate EDA technique divides the features in the dataset into two parts, namely numerical features and categorical features. Do an analysis of the category features first.

![categoricalfeature_status](https://user-images.githubusercontent.com/123156703/215409307-0c087496-0aca-443b-8895-a2df3f44bd8d.png)

Figure 5. Categorical Features


Based on the conclusion of the description of the variables, there are 2 categories in the number of features authorized more than posted.


![univariatenumericalfeatures](https://user-images.githubusercontent.com/123156703/215409320-0e960c8a-7340-4809-89c4-edc9bd5d5941.png)
Figure 6. Numerical Features


Meanwhile, in the numerical features category, the increase in amount is proportional to the decrease in the number of samples. We can see this clearly from the "amount" histogram, which graphs decrease as the number of samples (x-axis) increases.

* EDA Multivariate Analysis
Multivariate EDA shows the relationship between two or more variables in the data. Multivariate EDA which shows the relationship between two variables is commonly referred to as bivariate EDA. Next, perform data analysis on categorical and numeric features.

In the 'status' feature, it has a fairly large number of differences, namely having a difference of 17 amounts

on the 'amount' feature, the most range is between 60-70 transactions

on the 'account' feature the highest value reaches 70 transactions

in the 'currency' feature there is only one AUD variable which has a total amount of 29

in the 'long_lat' feature the total amount reaches 70

on the 'txn_description' feature the highest total number of transactions on PHONE BANK

in the 'merchant_id' feature, there is a balance between each transaction and the number of merchants

on the 'first_name' feature for a stable number of transactions

on the 'date' feature, the most number of transactions is above 40

In the 'gender' feature, the difference in the number of transactions for gender F and M is not much different

in the 'merchant_suburb' feature, it has a stable amount with rising and falling

in the 'merchant_state' feature the number of transaction usage is almost the same

the 'transaction_id' feature has a stable number of x, y variables when there is an increase and a decrease

in the 'transaction_id' feature, the highest amount is above 100

the 'country' feature only has one variable, namely the variable with the name of the Australian region which has a total number of transactions of 28-29.

on the 'customer_id' feature the highest transaction value reaches 70

on the 'merchant_long_lat' feature has a stable number of transactions

in the 'movement' feature there is only a debit variable which is almost 30 amounts


![corelasi](https://user-images.githubusercontent.com/123156703/215409311-1a11bad3-cb07-4c34-827d-1fad51d5e4bd.png)

Figure 7. Multivariate Analysis Correlation


An explanation of the correlation relationship between features. The correlation coefficient ranges between -1 and +1. Measures the strength of the relationship between two variables and their direction (positive or negative). Regarding the strength of the relationship between variables, the closer the value is to 1 or -1, the stronger the correlation. Meanwhile, the closer the value is to 0, the weaker the correlation. On the correlation graph, if we observe, the features 'age', 'balance,' amount 'have a fairly close correlation score. Meanwhile, the very low correlation only reached 0.06.

## Data Preparation
Data preparation is an important stage in the process of developing a machine learning model. This is the stage where we carry out the transformation process on the data so that it becomes a form suitable for the modeling process. In this section there are 4 stages of data preparation, but we will only use 3 according to needs, namely: Category feature encoding, Dataset division with the train_test_split function from the sklearn library, Standardization.
* Category feature encoding.
To carry out the process of encoding category features, one of the common techniques is the one-hot-encoding technique. The scikit-learn library provides this function to get appropriate new features so that it can represent categorical variables. This process functions to convert categorical variables into numeric variables.
* Dataset sharing with the train_test_split function from the sklearn library
Dividing the dataset into training data (train) and test data (test) is what we have to do before creating a model. We need to retain some of the existing data to test how well the model generalizes to the new data. Note that any transformations we perform on data are also part of the model. Because the test data (test set) acts as new data, we need to do all the transformation processes in the training data. This is the reason why the initial step is to split the dataset before doing any transformations. The goal is that we don't pollute the test data with the information we get from the training data. The proportion of training and test data sharing is usually 90:10.
 Result:
 
Table 1. Proportion of distribution of training data and test data


 |Total # of sample in whole dataset: 9054 |
    | ------ | ------ |
    |Total # of sample in train dataset: 8148 |
    | Total # of sample in test dataset: 906 |
    
* Standardization.
Standardization is the most commonly used transformation technique in the modeling preparation stage. For numeric features, we will not perform transformations with one-hot-encoding as for category features. We will use the StandardScaler technique from the Scikitlearn library,
StandardScaler performs the feature standardization process by subtracting the mean (average value) and then dividing it by the standard deviation to shift the distribution. The StandardScaler generates a distribution with a standard deviation of 1 and a mean of 0. About 68% of the values ​​will be between -1 and 1.

## Modeling
In this stage, we will develop a machine learning model with three algorithms. Then, we will evaluate the performance of each algorithm and determine which algorithm gives the best predictive results. The three algorithms that we will use include:
* K-Nearest Neighbor*
* Random Forest*
* Boosting Algorithm*


* K-Nearest Neighbor
KNN is a relatively simple algorithm compared to other algorithms. The KNN algorithm uses 'feature similarity' to predict the value of each new data. In other words, each new piece of data is assigned a value based on how similar it is in the training set. KNN works by comparing the distance of one sample to another training sample by selecting a number of k nearest neighbors (where k is a positive number). Well, that's why this algorithm is called *K-nearest neighbor* (a number of k nearest neighbors). KNN can be used for classification and regression cases. Although the KNN algorithm is easy to understand and use, it has disadvantages when faced with a large number of features or dimensions. This problem is often referred to as the *curse of dimensionality* (dimensional curse). Basically, this problem arises when the number of samples increases exponentially along with the number of dimensions (features) in the data. So, if you use the KNN model, make sure the data used has relatively few features. We use k = 10 neighbors and the Euclidean metric to measure the distance between points. At this stage we only train the training data and save the testing data for the evaluation stage which will be discussed in the Model Evaluation Module.
* Random Forest
Random forest is a machine learning model that is included in the ensemble (group) learning category. What is an ensemble model? Simply put, it is a prediction model that consists of several models and work together. The idea behind the ensemble model is that a group of models work together to solve a problem. Thus, the success rate will be higher than the model that works alone. In the ensemble model, each model must make predictions independently. Then, the predictions from each of these ensemble models are combined to make the final prediction.
The following are the parameters used:

    a. n_estimator: number of trees (trees) in the forest. Here we set n_estimator=50.
    
    b. max_depth: the depth or length of the tree. It is a measure of how much the tree can split (splitting) to divide each node into the desired number of observations max_depth=16.
    
    c. random_state: used to control the random number generator used random_state=55.
    
    d. n_jobs: the number of jobs that are used in parallel. It is a component to control threads or processes running in parallel. n_jobs=-1 means all processes are running in parallel.
    
* Boosting Algorithm
As the name suggests, boosting, this algorithm aims to improve performance or prediction accuracy. The trick is to combine several simple and weak models (*weak learners*) so as to form a strong model (strong ensemble learner). Boosting algorithms arose from the idea of ​​whether simple algorithms such as *linear regression* and *decision tree* could be modified to improve performance.
The following are the parameters used in the code snippet:

a. learning_rate: the weight applied to each regressor in each boosting iteration process, learning_rate=0.05.

b. random_state: used to control the random number generator used random_state=55.


## Evaluation
Evaluating a regression model is actually relatively simple. In general, almost all metrics are the same. If the predictions are close to the true values, the performance is good. Meanwhile, if not, poor performance. Technically, the difference between the actual value and the predicted value is called an error. Thus, all metrics measure how small the error value is. The metric that we will use in this prediction is MSE or *Mean Squared Error* which calculates the sum of the average squared difference between the true value and the predicted value.
let's look at the prediction results using some values ​​from the test data.


The metric that we will use in this prediction is MSE or Mean Squared Error which calculates the sum of the average squared differences between the true value and the predicted value. MSE is defined in the following equation

![formula](https://user-images.githubusercontent.com/123156703/215319673-c374edc2-1c5f-486f-93fc-3ff127a2d6f9.jpeg)

Figure 8. Formula for calculating MSE


Information:

N = number of datasets

yi = true value

y_pred = predicted value



When calculating the Mean Squared Error value on the train and test data, we divide it by the value 1e3. It is intended that the mse value is on a scale that is not too large. The evaluation results on the training data and test data are as follows:



Table 2. Evaluation results of training data and test data
|  | train	 | test |   
| ------ | ------ | ------ |
|KNN | 0.260764 | 0.476869
| RF | 0.239543 | 0.352439
|Boosting | 0.354629 | 0.369491


To test it, make a prediction using some prices from the test data.
Table 3. Price prediction test results from the datasheet


| y_true | 9.93	 |
| ------ | ------ |
|prediksi_KNN | 22.0 |
| prediksi_RF | 26.8 |
|prediksi_Boosting | 29.2 |


## Conclusion

![evaluasimodel](https://user-images.githubusercontent.com/123156703/215409312-7d4e8582-4dc8-4b08-8418-a75edd84c608.png)

Figure 9. Predictions for each model


It can be seen that the prediction with K-Nearest Neighbor (KNN) gives the closest result. To improve performance, do the same thing (setting parameters) for all algorithms used. In addition, perform parameter optimization by applying the Grid Search technique.


## Reference
[1]	M. Sathye, “Internet Banking in Australia,” SSRN Electron. J., pp. 1996–1998, 2005, doi: 10.2139/ssrn.38222.
[2]	J. Hirst and M. J. Taylor, “The internationalisation of Australian banking: further moves by the ANZ,” Aust. Geogr., vol. 16, no. 4, pp. 291–295, 1985, doi: 10.1080/00049188508702886.



