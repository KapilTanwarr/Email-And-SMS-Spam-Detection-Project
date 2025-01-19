# Email-And-SMS-Spam-Detection-Project
This project is a machine learning pipeline to detect spam messages using the SMS Spam Collection Dataset. The dataset contains a collection of SMS messages labeled as either spam or ham (non-spam). The goal of this project is to preprocess the text data, build a classification model, and evaluate its performance. 

# Table of Contents

Data Cleaning

Exploratory Data Analysis (EDA)

Text Preprocessing

Model Building

Evaluation

Deployment

# 1. Data Cleaning

The dataset contained 5572 entries with 5 columns, of which 3 were irrelevant and dropped.

Duplicate entries (403) were removed, leaving 5169 unique entries.

Columns were renamed for clarity:

v1 -> target

v2 -> text

The target variable was encoded (0: ham, 1: spam).

# 2. Exploratory Data Analysis (EDA)

Imbalanced Data: 87% ham and 13% spam messages.

Added new features:

Number of characters

Number of words

Number of sentences

# Insights:

Spam messages are longer on average (137 characters) compared to ham (70 characters).

Visualization included:

Pie chart showing class distribution.

Histograms for message lengths.

Word clouds for spam and ham messages.

# 3. Text Preprocessing

Steps:

Lowercased all text.

Tokenized into words.

Removed stopwords, punctuation, and non-alphanumeric characters.

Stemming using Porter Stemmer.

Transformed text saved in a new column: transformed_text.

# 4. Model Building

Prepared spam and ham corpora for analysis.

Word frequency visualizations using bar plots.

Upcoming steps include:

Train-test split.

Building machine learning models (e.g., Naive Bayes, Logistic Regression).

Comparing model performances.

# 5. Evaluation

Metrics to be used:

Accuracy

Precision

Recall

F1-score

# 6. Deployment

Deployment will include:

A web application to classify SMS messages.

Deployment platform: Streamlit or Flask (to be finalized).

# Tools and Libraries Used

Python (pandas, NumPy, matplotlib, seaborn, nltk)

Scikit-learn

WordCloud

How to Run

Clone the repository.

Install required libraries:

pip install -r requirements.txt

Run the notebook or script to preprocess data and train models.

(Optional) Deploy the model to a web application.

# Acknowledgments

SMS Spam Collection Dataset creators.

Open-source Python libraries used in this project.

# Future Enhancements

Use advanced NLP techniques like TF-IDF and word embeddings.

Experiment with deep learning models (e.g., LSTMs, Transformers).

Improve deployment UI and integrate with real-world SMS APIs.

