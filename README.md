## Documentation for Code and Insights

This documentation outlines the code structure and insights derived from the dataset concerning credit card transactions and customer behavior. The methodology employed aims to preprocess data, train a machine learning model, and evaluate its performance effectively.

### How to run?

1. Add data by creating "raw" folder in "data", which contains "dev_data.csv" and "val_data.csv"
2. Then, run the "dataloader.py" script in "src/data/dataloader.py"
   1. You can tweak with different settings and other preprocessing steps you want
3. Then run the "lgb_train_and_predict.py" file, which will run the hyperparameter, training and prediction modules.

This project was made for the Convolve4.0 competition where we had to determine the probability of a person defaulting on their credit.

**NOTE:** I was limited by my hardware (and the cost of training models on cloud) T-T
So I haven't properly tuned any of the hyperparameters and just random guessed most of them based on other people's work. I recommend tuning the hyperparameters first, and then running the training and prediction modules by commenting them out, if you face the same problems as me.

In the future I will come back to this repo to make it make it more approachable and easy to work with.

### Insights Overview

The dataset includes various attributes related to customer behavior, which can be categorized into four main types:

1. On-us attributes
      1. On-us attributes refer to data points that are internal to the bank issuing the credit card. 
      2. These attributes are derived from the bank's own records and systems, without relying on external sources like credit bureaus.
         1. An example is the credit limit, which is the maximum amount of money a customer is allowed to borrow on their credit card.
         2. It is determined by the bank based on factors such as the customer's income, repayment history, and creditworthiness.
      3. The credit limit directly influences customer behavior and default risk:
         1. Customers with higher limits may have greater spending capacity but also pose a higher risk if they overextend themselves.
         2. Customers with lower limits are less likely to default but may feel constrained in their spending.
      4. It also impacts credit utilization, which is a critical factor in assessing financial health. High utilization (spending close to the limit) can indicate financial stress and increase default risk.
         1. If a customer has a credit limit of ₹1,00,000 and consistently uses ₹90,000 or more, it suggests high reliance on credit, which could be a red flag for potential default.

2. Transaction level attributes
      1. Transaction-level attributes capture detailed information about a customer's spending patterns and habits. 
      2. These are derived from the transactions made using the credit card.
         1. Number of Transactions: Total count of transactions made within a specific period.
         2. Rupee Value of Transactions: The monetary value of transactions.
         3. Merchant Categories: Spending categorized by merchant type (e.g., groceries, fuel, luxury goods, travel).
      3. Transaction-level attributes provide insights into a customer's lifestyle, spending behavior, and financial discipline:
         1. Frequent small-value transactions may indicate controlled spending habits.
         2. Large-value transactions at luxury merchants may suggest higher risk if not matched with income levels.
      4. These attributes can also highlight changes in behavior that might signal financial distress (e.g., sudden spikes in spending or increased reliance on essential categories like groceries).
         1. If a customer suddenly increases their transaction volume or spends heavily on discretionary items like luxury goods while nearing their credit limit, it might indicate risky financial behavior.

3. Bureau tradeline level attributes -> Bureau
      1. product holdings
         1. Number and types of active credit products held by the customer
         2. credit cards, home loans, personal loans, auto loans
         3. A higher number of active accounts -> higher financial obligations and strain on repayment capacity
      2. historical delinquencies 
         1. missed payments or overdue amounts on any credit products 
         2. strong indicators of future defaults
      3. Credit Utilization
         1. The ratio of the outstanding balance to the total available credit limit across all accounts
         2. High utilization -> financial stress or over-leverage
      4. Account Age
         1. Longer account -> stable and responsible credit behavior

4. Bureau enquiry level attributes -> Bureau enquiry
      1. Personal Loan Enquiries in the Last 3 Months
         1. Number of personal loan applications made by the customer in the last three months.
         2. High enquiries -> aggressive borrowing or financial stress
      2. Total number of hard inquiries made in recent periods
         1. Frequent inquiries -> actively seeking credit
         2. sign of over-leverage or an impending liquidity crunch.
      3. Inquiry Type
         1. The type of product for which inquiries were made (e.g., personal loans, car loans, mortgages).
         2. Different inquiry types -> varying levels of risk
         3. frequent payday loan inquiries -> higher risk compared to mortgage inquiries.

The dataset is highly imbalanced, with 95,434 instances of `bad_flag` = 0 and only 1,372 instances of `bad_flag` = 1.

### Methodology

The following steps outline the workflow used to process the dataset and build a predictive model:

#### Data Preprocessing
The dataset has four types of features, onus_attribute_*, transaction_attribute_* bureau_attribute_*, and bureau_enquiry_*

On visual inspection, onus_attribute_1 tells us the credit limit of a customer in the bank. From this, few of transaction_attribute_* tells the transactions that happened by the account. 
Thus, if onus_attribute_1 is NaN, the following transaction_attribute_* are also NaN.

1. **Data Loading**:

   - Load raw datasets (`dev_data.csv` and `val_data.csv`) into Parquet format for efficient processing.

2. **Feature Engineering**:

   - Removing features with zero sums or those with same with no distict values.
   - Create new features by building relations between the feature groups.
   - Clipping outliers for better data structure.

3. **Creating Summary Features**:

   - Aggregate features such as means, sums, counts, and ratios from existing attributes.

4. **Power Transformation**:
   - Using power transformation for skewness.

### Model Training

1. **Model Configuration**:

   - LightGBM model parameters including objectives, metrics (AUC and binary_logloss), and hyperparameters for tuning.

2. **Hyperparameter Tuning**:

   - Utilizing Optuna for optimizing hyperparameters through cross-validation with stratified K-Folds.

3. **Training the Model**:

   - Training the LightGBM model on the processed training data while monitoring performance metrics such as AUC, precision, recall, and F1-score.

4. **Model Evaluation**:

   - Evaluating the model on a separate test set (derived from the given dev set) to ensure it generalizes well to unseen data.

5. **Prediction**:
   - Generate predictions on the test set and save results for further analysis.

### Code Structure

The code is organized into several modules for clarity and modularity:

- **Data Manipulation Module**: Contains functions for preprocessing steps including denoising, clipping outliers, feature creation, and handling missing values.
- **Dataset Split Module**: Responsible for splitting the dataset into training and testing sets with stratification.
- **Model Training Module**: Implements functions to train the LightGBM model and tune hyperparameters.
- **Prediction Module**: Handles predictions based on trained models and evaluates performance metrics.

### Conclusion

This documentation captures every aspect of the code developed to analyze customer behavior based on credit card history. By applying systematic preprocessing techniques followed by robust modeling strategies, we aim to derive insights that can help in predicting customer default risks effectively. The methodology ensures that all necessary steps are taken to handle imbalances in the dataset while maximizing predictive accuracy through careful evaluation of model performance metrics.
