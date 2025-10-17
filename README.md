📧 Spam-Ham-Email-Detector

A machine learning project that detects whether a message is Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Logistic Regression.

🧠 Overview

This project demonstrates how to build a Spam Detection System using Python and Machine Learning techniques.
It involves:

Cleaning and preprocessing text data

Converting text into numerical form using TF-IDF

Training a Logistic Regression model

Evaluating the model’s accuracy and performance

Visualizing spam vs ham predictions

🗂️ Dataset

The dataset used is spam_ham_3000.csv, which contains:

label → Indicates whether the message is ham or spam

message → The text message content

Example:

label	message
ham	Hello! How are you?
spam	Congratulations! You’ve won a prize!
⚙️ Technologies Used

Python 🐍

Pandas → For data handling

Regex (re) → For text cleaning

Matplotlib → For visualization

scikit-learn → For ML model building (TF-IDF & Logistic Regression)

🧩 Steps in the Project
1️⃣ Load and Explore Dataset

The dataset is loaded using pandas.read_csv() to inspect and understand message distribution.

2️⃣ Text Cleaning

All messages are converted to lowercase, punctuation and numbers are removed for better model accuracy.

3️⃣ Label Encoding

‘ham’ → 0
‘spam’ → 1

4️⃣ Data Splitting

The dataset is split into training (80%) and testing (20%) sets using train_test_split().

5️⃣ Text Vectorization

TF-IDF Vectorizer converts text into numerical values that can be used by the ML model.

6️⃣ Model Training

A Logistic Regression model is trained on the TF-IDF vectors.

7️⃣ Model Evaluation

Model accuracy and performance are measured using:

Accuracy Score

Classification Report

Confusion Matrix

8️⃣ Visualization

A bar chart shows how many messages are predicted as Spam or Ham.

📊 Results

Displays accuracy score (e.g., Accuracy: 0.97)

Shows spam and ham message counts

Generates a clean visualization for predictions


