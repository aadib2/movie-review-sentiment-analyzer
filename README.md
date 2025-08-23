# 🎬 Movie Reviews Sentiment Analysis
A machine learning web app for analyzing movie review sentiment using natural language processing and explainable AI. 

Built with scikit-learn, NLTK, SHAP, and Streamlit, this project classifies movie reviews as positive or negative with detailed explanations of model predictions.

Originally created as the final project for the Cornell Tech ML Foundations Certification through Break Through Tech, expanded with NLP techniques, explainable AI features, and web app. More specifics can be found below.

---

## 🚀 Live Demo Site
- 👉 [Streamlit App](https://imdb-sentiment-analyzer-btt.streamlit.app/)
- 🧠 Model accuracy: 85% on test dataset
- 📊 AUC Score: 92%
- 📁 Dataset: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) - **only a sample of 6k reviews used for project**

## 🧠 Project Overview

- **Goal**: Classify movie reviews into sentiment categories:
    - Positive (good/favorable reviews)
    - Negative (bad/unfavorable reviews)

- **Model**: Logistic Regression with TF-IDF vectorization
- **Key Features:**
    - Text preprocessing pipeline using NLTK library & regex cleaning
    - TF-IDF feature extraction
    - Modeling using Logistic Regression with hyperparameter tuning using GridSearchCV, trained on sample of 4500 reviews
    - Model evaluation using various metrics like accuracy, confusion matrix & ROC-AUC curve (sklearn)
    - SHAP explainability for model interpretations: Interactive waterfall plots showing word contributions
    - Interactive web app showcasing predictive ability + model interpretability (Streamlit)

## 🛠️ Tech Stack
| Tool                            | Purpose                          |
|---------------------------------|----------------------------------|
| Python                          | Core programming language             
| scikit-learn                    | ML models + metrics
| NLTK                            | Natural Language Processing
| SHAP                            | Model explainability and interpretations
| Streamlit                       | Interactive web app 
| pandas / numpy                  | Data manipulation & analysis, vector operations
| matplotlib                      | Data visualization
| regex / string                  | Text preprocessing and cleaning

---

## 📊 Model Performance

- Accuracy: 85%
- AUC Score: 92%
- Precision: 82% (Positive)
- Recall: 87% (Positive)
- Confusion Matrix: insert HERE
- ROC Curve: insert HERE

---

## 🎯 Key Features

- **Interactive Review Analysis**: Enter any movie review and get instant sentiment predictions
- **Explainable AI**: SHAP waterfall plots show exactly which words influenced the prediction and by how much
- **Model Interpretability**: Understand why the model made specific decisions, especially for ambiguous reviews
- **Probability Visualization**: See confidence scores for both positive and negative predictions
- **Text Preprocessing Pipeline**: Complete NLP preprocessing


📍 To run locally:

Clone the repository:

`git clone https://github.com/your-username/movie-review-sentiment-analyzer.git`

`cd movie-review-sentiment-analyzer`

Create and activate virtual environment:

`python -m venv venv`

`source venv/bin/activate  # On Windows: venv\Scripts\activate`

Install dependencies:

`pip install -r requirements.txt`

Run the Streamlit app:

`streamlit run app.py`

Open your browser and go to http://localhost:8501


## 📁 Project Structure

TO BE UPDATED

🔍 Acknowledgements:

- IMDB Movie Reviews Dataset contributors
- Cornell Tech and Break Through Tech for the ML Foundations Certification program
- SHAP library developers for explainable AI tools
- Streamlit community for deployment tutorials

