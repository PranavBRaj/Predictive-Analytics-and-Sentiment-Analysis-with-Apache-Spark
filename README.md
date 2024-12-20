# Predictive-Analytics-and-Sentiment-Analysis-with-Apache-Spark
This project demonstrates the application of machine learning techniques for two real-world scenarios using Apache Spark: Sentiment Analysis on Customer Feedback and Price Prediction for Travel Bookings. The project leverages Spark's MLlib to process large-scale data efficiently, showcasing the power of distributed computing for scalable and robust machine learning workflows.

Key Features
Sentiment Analysis on Customer Feedback

Objective: Analyze customer feedback to classify sentiments as positive or negative.
Pipeline Stages:
Text Preprocessing: Tokenization, stop word removal, and feature extraction using TF-IDF.
Modeling: Logistic Regression for binary classification.
Implementation:
Preprocess feedback to extract meaningful features.
Train and test a sentiment analysis model using labeled feedback data.
Generate predictions with probabilities for each input feedback.
Output: A pipeline capable of classifying customer feedback with sentiment labels.
Travel Pricing Prediction

Objective: Build a regression model to predict travel pricing based on temporal and booking data.
Pipeline Stages:
Feature Engineering: Combine day, month, year, and bookings into a single feature vector.
Modeling: Linear Regression to predict pricing.
Implementation:
Assemble features from historical travel data.
Train a regression model on pricing data.
Test the model with new booking data to predict future prices.
Output: A predictive model to forecast travel prices based on key variables.
