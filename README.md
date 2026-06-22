# 🚀 YouTube Viral Channel Prediction

## 📌 Project Overview

This project is a Machine Learning-based web application that predicts whether a YouTube channel has the potential to become viral. The model analyzes various channel statistics and growth indicators to classify channels as **Viral** or **Not Viral**.

The application is built using **Python**, **Scikit-Learn**, and **Streamlit**, providing an interactive interface where users can enter channel information and receive instant predictions.

---

## 🎯 Problem Statement

With millions of YouTube channels competing for attention, identifying channels with strong viral potential can be challenging. This project uses historical channel data and machine learning techniques to predict whether a channel is likely to achieve viral growth.

---

## 📊 Dataset Features

The model uses the following features:

* Subscribers
* Video Views
* Uploads
* Country
* Channel Type
* Created Year
* Population
* Urban Population
* Channel Age
* Views Per Upload

### Target Variable

* **Viral Channel (0/1)**

  * 1 = Viral
  * 0 = Not Viral

---

## ⚙️ Machine Learning Workflow

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Label Encoding of Categorical Variables
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Streamlit Deployment

---

## 🤖 Models Evaluated

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Decision Tree
* Support Vector Machine (SVM)
* Random Forest

### Best Model

**Random Forest Classifier**

Performance:

* Accuracy: **79.04%**
* F1 Score: **79.66%**

---

## 🎨 Web Application Features

* Modern Dark-Themed UI
* Interactive Sidebar Inputs
* Real-Time Predictions
* Virality Probability Score
* Animated Dashboard Components
* Responsive Design

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* HTML/CSS

---

## 📈 Future Improvements

* Integration of real-time YouTube API data
* Advanced feature engineering
* Hyperparameter optimization
* Deep Learning-based prediction models
* Enhanced analytics dashboard

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run yotube.py
```

---

## 📷 Project Preview

A Streamlit-based dashboard that predicts YouTube channel virality using machine learning and provides confidence scores with an interactive user interface.

---

## 👨‍💻 Author

Tarun Sharma

Master of Computer Applications (MCA)

Machine Learning & Data Science Enthusiast
