# Customer Segmentation Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-red)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-yellow)

## Introduction

This project focuses on **customer segmentation** using **K-Means Clustering**, an unsupervised machine learning algorithm. The dataset contains customer transaction behaviors, including features such as `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and more. The goal is to group customers into distinct clusters based on their spending and payment patterns, enabling businesses to tailor marketing strategies and improve customer engagement.

### Dataset Features
The dataset includes the following key features:
- **CUST_ID:** Unique identifier for each customer.
- **BALANCE:** Outstanding balance on the customer's account.
- **BALANCE_FREQUENCY:** Frequency of balance updates (0 to 1).
- **PURCHASES:** Total purchase amount.
- **ONEOFF_PURCHASES:** Total one-off purchase amount.
- **INSTALLMENTS_PURCHASES:** Total installment purchase amount.
- **CASH_ADVANCE:** Total cash advance amount.
- **PURCHASES_FREQUENCY:** Frequency of purchases (0 to 1).
- **ONEOFF_PURCHASES_FREQUENCY:** Frequency of one-off purchases (0 to 1).
- **PURCHASES_INSTALLMENTS_FREQUENCY:** Frequency of installment purchases (0 to 1).
- **CASH_ADVANCE_FREQUENCY:** Frequency of cash advances (0 to 1).
- **CASH_ADVANCE_TRX:** Number of cash advance transactions.
- **PURCHASES_TRX:** Number of purchase transactions.
- **CREDIT_LIMIT:** Customer's credit limit.
- **PAYMENTS:** Total payments made by the customer.
- **MINIMUM_PAYMENTS:** Minimum payments made by the customer.
- **PRC_FULL_PAYMENT:** Percentage of full payments made.
- **TENURE:** Tenure of the customer (in years).
