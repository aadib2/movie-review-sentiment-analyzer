# install dependencies (possibly store in requirements.txt)
import streamlit as st
import pickle
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import preprocess

# load in all necessary files
model_lr_best = pickle.load(open('log_reg_model.sav', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
explainer = pickle.load(open('shap_explainer.sav', 'rb'))

# sections of the webpage:

# title and description(explanation of project)
st.header("🎥 Movie Reviews Sentiment Analysis - Demo")
st.subheader("Created by Aadi Bery")

st.write("This project was originally created as the final project as a part of the Cornell Tech ML Foundations Certification through BTT but expanded further and built on with new techniques and tools to make it more complete.")
st.write("This web app allows you to input your own movie review and see whether the model predicts it to be a good or bad review, along with an explanation of the prediction using SHAP values (explainable AI!)")
st.markdown("This is acheived through a simple logistic regression model with accuracy is **~85%** and **auc score ~92%**. If you are interested I've listed more about this on the [Github README](https://github.com/aadib2/movie-review-sentiment-analyzer)")

# Add some example reviews users can click to try
st.write("**Try one of these examples to get started or input your own review!**")
col1, col2 = st.columns(2)

# --- RESET LOGIC: must be before st.text_area ---
if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

if st.button("Reset"):
    st.session_state["review_input"] = ""
    st.rerun()  # rerun to clear the text box immediately

# if buttons were pressed then prepopulate review
with col1:
    if st.button("😊 Positive Example"):
        st.session_state["review_input"] = "This movie was absolutely fantastic! Amazing acting and brilliant storyline."
        st.rerun()

with col2:
    if st.button("😞 Negative Example"):  
        st.session_state['review_input'] = "Terrible movie. Boring plot and awful acting. Complete waste of time."
        st.rerun()

# text input of review
review_input = st.text_area(
    "📝 Enter your movie review here!", 
    value=st.session_state['review_input'],
    key="review_input",
    placeholder="Type your movie review here...",
    height=150
)



# perform prediction pipeline
if(st.button('Predict') and review_input != ""):

    # we need to first preprocess this review (removing stopwords, punctuation, tokenize, etc.)
    cleaned_review = preprocess(review_input)

    # now vectorize this cleaned review using the tfidf vectorizer we fit earlier
    vectorized_review = tfidf_vectorizer.transform([cleaned_review])

    # make our prediction
    prediction = model_lr_best.predict(vectorized_review)
    prob = model_lr_best.predict_proba(vectorized_review)

    # retrieve new shap values to showcase prediction
    shap_value = explainer.shap_values(vectorized_review)

    # display information
    st.write('**Original Review:** ', review_input)
    st.write('**Processed Review:** ', cleaned_review)

    goodReview = True if prediction == 1 else False
    st.write('\nPrediction: Is this a good review? {}\n'.format(goodReview)) 

    prob_df = pd.DataFrame({
    'Sentiment': ['Negative', 'Positive'],
    'Probability': prob[0]
    })  
    st.bar_chart(prob_df.set_index('Sentiment'))
    

    # create the waterfall plot, showing it is a little bit different in streamlit
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title("SHAP Explanation for Prediction")

    explanation = shap.Explanation(
        values=shap_value[0],
        base_values=explainer.expected_value,
        data=vectorized_review.toarray()[0], # remember to convert to array since it is a sparse matrix
        feature_names=tfidf_vectorizer.get_feature_names_out()
    )

    shap.waterfall_plot(explanation, max_display=15, show=False) # display the top 15 features / 'words' which are contributing

    st.pyplot(fig)
    plt.clf()


