import spacy
import re
import numpy as np
import pickle
import streamlit as st
from PIL import Image

image = Image.open('images/love_wall.jpeg')
st.image(image, width=480)

# Load the spacy model
nlp = spacy.load('en_core_web_sm')

# Load the model
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer_w2v_model.pkl', 'rb'))

def remove_special_characters(string):
    # Use a regular expression to match all non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', string)

def preprocess_string(string):
    # Parse the string using the spacy model
    doc = nlp(string)

    # Tokenize the string
    tokenized_string = [token.text for token in doc]

    # Remove stop words from the string
    filtered_string = [word for word in tokenized_string if not nlp.vocab[word].is_stop]

    # Remove special characters from the string
    filtered_string = [remove_special_characters(word) for word in filtered_string]

    # Lemmatize the string
    lemmatized_string = [token.lemma_.lower() for token in doc if token.text in filtered_string]

    return lemmatized_string

def transform_text(tweet):
    # Preprocess the string
    clean_string = preprocess_string(tweet)

    # Join the preprocessed string into a single string
    transformed_string = ' '.join(clean_string)

    return transformed_string

def get_vector(vectorizer, tweet, vector_length):
    # Split the tweet into a list of words
    words = tweet.split()

    # Initialize the vector with zeros
    vector = np.zeros(vector_length)

    # Iterate over the words in the tweet
    for word in words:
        # If the word is in the vocabulary, add its vector to the total vector
        if word in vectorizer.wv:
            vector += vectorizer.wv[word]

    # Return the average vector
    return vector / len(words)

st.title("Hate speech via tweet Classifier")

input_tweet = st.text_area("Enter the tweet")


if st.button('Predict'):
    # 1. preprocess
    transformed_tweet = transform_text(input_tweet)
    # 2. vectorize
    vector_input = get_vector(vectorizer, transformed_tweet,300)
    # 3. predict
    result = model.predict(vector_input.reshape(1, -1))[0]
    # 4. Display
    if result == 1:
        st.header("This tweet appears to have hate speech")
    else:
        st.header("This tweet does not appear to have hate speech")

st.markdown(
"""
<style>
.reportview-container .main .block-container{{
    background-color: black;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<link href='https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;700&display=swap' rel='stylesheet'>

<style>
h1 {
    font-family: 'Roboto Slab';
    font-weight: 700;
    color: maroon;
}
</style>
""", unsafe_allow_html=True)

sentences = [
    "asians are disgusting",
    "black people are so trash",
    "I love mondays",
    "I hate Asians",
    "women should not be allowed to work because they are lazy",
    "immigrants should be banned",
    "women are retards",
    "I am so hateful towards asians",
    "I do not like any muslim",
    "trump is hateful"
]

# Use the `st.header` function to add a header
st.header("Here are some prompts of tweets to test the model")

# Use the `st.text` function to display the sentences in a text box
st.text("\n".join(sentences))

prompts_text = "The prompts for testing are either auto generated or coming from a collection of offensive tweets from the test.csv file found here: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=test.csv. So please don't cancel me!"
difficulty_text = "When you try to come up with your own prompt and get incorrect results, this may be due to a number of factors, some of which are:"
ambiguity_text = "1. Ambiguity in language: Hate speech often relies on sarcasm, irony, or double meanings, making it difficult for a machine to correctly interpret the intent behind a statement. Additionally, hate speech can be disguised using euphemisms and coded language, which can make it difficult to detect."
data_text = "2. Limited data set: In order to train a machine learning model to accurately classify hate speech, a large and diverse dataset is needed. However, hate speech is often underrepresented in publicly available data sets and can be difficult to collect in an ethical manner, which makes it challenging to train models that perform well on this task."
language_text = "3. Rapidly evolving language: The way language is used on social media platforms is constantly evolving, making it difficult to create a model that can generalize to new and emerging forms of hate speech. Additionally, slang, dialects, and regional variations in language can also make it difficult to classify hate speech. This is further complicated by the fact that hate speech is not only restricted to text and can be found in different forms like images and videos, which also require different treatment methods."
capital_text = "4. Correct capitalising is also important, for example: 'i hate asians' may not work but 'I hate Asians' works so when you input a text, check for any capitals and correct grammar"
disclaimer_text = "The results of this application are for informational purposes only and should not be used as a substitute for professional advice."

st.header("Disclaimer")
st.markdown(prompts_text)
st.markdown(difficulty_text)
st.markdown(ambiguity_text)
st.markdown(data_text)
st.markdown(language_text)
st.markdown(capital_text)
st.markdown(disclaimer_text)