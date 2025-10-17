import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("spam_ham_3000.csv")

# Clean the messages
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Convert labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Count spam and ham in predictions
spam_count = sum(y_pred)
ham_count = len(y_pred) - spam_count

# Plot spam vs ham and show in Colab
plt.figure(figsize=(6, 4))
plt.bar(['Ham', 'Spam'], [ham_count, spam_count], color=['green', 'red'])
plt.title('Predicted Email Categories')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print results
print("Accuracy:", round(accuracy, 4))
print("Spam Emails Detected:", spam_count)
print("Ham Emails Detected:", ham_count)
