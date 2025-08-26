# 📧 Email/SMS Spam Detection using Machine Learning

This repository contains a machine learning project that classifies text messages and emails into Spam or Ham (Not Spam).  
The goal is to build a reliable spam detection system using Natural Language Processing (NLP) and Machine Learning algorithms.  

Spam detection is crucial in real-world applications such as email filtering, SMS filtering, fraud detection, and cybersecurity systems.  

# 🎯 Objectives
- Build a robust spam detection system using ML.
- Apply text preprocessing (tokenization, stopword removal, stemming/lemmatization).
- Extract features using Bag-of-Words (BoW) and TF-IDF.
- Train and evaluate multiple ML algorithms.
- Achieve high accuracy and minimize false positives.

# 📂 Dataset
- SMS Spam Collection Dataset (Kaggle Dataset)
- Dataset contains 5,572 messages labeled as:
- ham → Legitimate messages
- spam → Unwanted or promotional messages


# 🔄 Project Workflow
- **1. Data Collection** → Import dataset (CSV/TSV file).
- **2. Data Preprocessing** →
  - Lowercasing text
  - Removing punctuation, numbers, and stopwords
  - Tokenization & Lemmatization/Stemming

- **3. Feature Engineering** → Convert text into numerical format using:
  - Bag-of-Words (BoW)
  - TF-IDF Vectorization
- **4. Model Training** → Train multiple models:
  - Multinomial Naïve Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- **5. Model Evaluation** → Evaluate with:
  - Accuracy, Precision
  - Confusion Matrix
- **6. Results & Insights** → Select the best model.


# 🛠 Technologies Used
- **Language:** Python
- **Libraries:**
  - numpy, pandas → Data processing  
  - nltk, re, scikit-learn → NLP + ML  
  - matplotlib, seaborn → Data visualization  
- **Tools:** jupyter notebook → Experimentation  

# 📊 Exploratory Data Analysis (EDA)
- Distribution of Ham vs Spam messages
- Most frequent words in Spam vs Ham (WordClouds)
- Message length distribution
- Correlation heatmap of features


# 🤖 Models Implemented

- ✅ Multinomial Naïve Bayes (MultinomialNB Algorithm)
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)
- ✅ Random Forest Classifier
- ✅ K-Nearest Neighbors (KNN)
- ✅ Random Forest (RF)
- ✅ Decision Tree (DT)
- ✅ AdaBoostClassifier
- ✅ BaggingClassifier (BgC)
- ✅ ExtraTreesClassifier (ETC)
- ✅ GradientBoostingClassifier (GBDT)
- ✅ XGBClassifier (xgb)

📌 Best Model: Multinomial Naïve Bayes with ~97% accuracy and precision 1.00.

# 🏆 Results
| Model                             | Accuracy | Precision 
| --------------------------------  | -------- | --------- 
| Naïve Bayes                       | 97.09%   | 1.00     
| Logistic Regression               | 95.55%   | 0.96      
| SVM                               | 97.58%   | 0.97      
| Random Forest                     | 94%      | 0.92      
| KNN                               | 90.52%   | 1.00   
| Random Forest (RF)                | 97.38%   | 0.98 
| Decision Tree (DT)                | 93.23%   | 0.83   
| AdaBoostClassifier                | 92.16%   | 0.82
| BaggingClassifier (BgC)           | 95.84%   | 0.86
| ExtraTreesClassifier (ETC)        | 97.48%   | 0.97
| GradientBoostingClassifier (GBDT) | 95.06%   | 0.93
| XGBClassifier (xgb)               | 96.80%   | 0.94  



# 🚀 Future Enhancements
- Deploy as a Flask/Streamlit web app
- Implement Deep Learning models (LSTM, BERT)
- Real-time spam detection API

