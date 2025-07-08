# Flipkart Customer Satisfaction (CSAT) Prediction

This project focuses on predicting customer satisfaction using real-world customer service interaction data from Flipkart. The goal is to classify whether a customer is satisfied or dissatisfied based on structured features and customer remarks.

## Project Objective

To build a machine learning model that can help identify dissatisfied customers early, enabling Flipkart to take proactive measures and improve overall service quality.

## Dataset Overview

- Over 85,000 customer service records
- Features include:
  - Product and issue categories
  - Channel of interaction
  - Agent shift and customer city
  - Item price and tenure bucket
  - Customer remarks (textual feedback)
- Target variable: CSAT Score (1 to 5), converted into binary:
  - 1–3 → Not Satisfied (0)
  - 4–5 → Satisfied (1)

## Methodology

1. **Data Cleaning & Preprocessing**
   - Removed redundant and ID columns
   - Handled missing values using appropriate strategies
   - Label encoded categorical variables
   - Extracted features from timestamp and remarks

2. **Exploratory Data Analysis**
   - Identified patterns in shift, price, product category, and satisfaction
   - Plotted CSAT score distribution, remark lengths, and channel types

3. **Model Training**
   - Models Tested: Logistic Regression, Random Forest, Gradient Boosting
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
   - Final Model: Gradient Boosting Classifier (Accuracy: 0.75)

4. **Hyperparameter Tuning**
   - Used GridSearchCV to tune `n_estimators`, `max_depth`, `learning_rate`, and more
   - Applied 5-fold cross-validation

5. **Evaluation**
   - Confusion matrix and classification report
   - Metric score charts (bar graphs)

## Technologies Used

- Python
- Google Colab
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Results

- Final Model: Gradient Boosting Classifier
- Accuracy: 0.75
- Best-performing model after hyperparameter tuning
- Successfully identified dissatisfied customers with good recall and F1-score

## Business Impact

This model can help Flipkart:
- Proactively follow up with unhappy customers
- Improve agent assignment and service quality
- Optimize operational timing (e.g., shift and category targeting)

## Author

This project was completed as part of an internship submission. It includes end-to-end ML implementation along with EDA, modeling, evaluation, and interpretation.
