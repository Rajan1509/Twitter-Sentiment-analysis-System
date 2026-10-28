import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
stop = stopwords.words('english')

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.title("Twitter Sentiment Analysis Dashboard")
st.write("Machine Learning Classification with Streamlit + Sentiment140 Dataset")

# ----------------------------------------------------
# FILE UPLOADER
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload the Sentiment140 CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1', header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]

    st.subheader("Dataset Preview")
    st.write(df.head())

    # ------------------------------------------------
    # CLEANING FUNCTION
    # ------------------------------------------------
    def clean_tweet(tweet):
        tweet = re.sub(r"http\S+", "", tweet)       # remove URLs
        tweet = re.sub(r"@\w+", "", tweet)          # remove mentions
        tweet = re.sub(r"[^a-zA-Z ]", "", tweet)    # keep only letters
        tweet = tweet.lower()
        words = [w for w in tweet.split() if w not in stop]
        return " ".join(words)

    df["clean_text"] = df["text"].apply(clean_tweet)

    # ------------------------------------------------
    # SENTIMENT DISTRIBUTION
    # ------------------------------------------------
    st.subheader("Sentiment Distribution")
    df['sentiment'] = df['target'].replace({0: "Negative", 4: "Positive"})
    sentiment_counts = df['sentiment'].value_counts()

    st.bar_chart(sentiment_counts)

    # PIE CHART
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

    # ------------------------------------------------
    # WORDCLOUDS
    # ------------------------------------------------
    st.subheader("Wordcloud - Positive Tweets")
    pos_text = " ".join(df[df['sentiment'] == "Positive"]['clean_text'])
    pos_wc = WordCloud(width=800, height=400).generate(pos_text)
    st.image(pos_wc.to_array())

    st.subheader("Wordcloud - Negative Tweets")
    neg_text = " ".join(df[df['sentiment'] == "Negative"]['clean_text'])
    neg_wc = WordCloud(width=800, height=400).generate(neg_text)
    st.image(neg_wc.to_array())

    # ------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_text"]).toarray()
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    st.subheader("Model Evaluation Metrics")
    st.text(classification_report(y_test, pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    st.write(cm)

    # ------------------------------------------------
    # LIVE TWEET PREDICTION
    # ------------------------------------------------
    st.subheader("Test Your Own Tweet")
    user_input = st.text_area("Enter a tweet to analyze sentiment:")

    if st.button("Predict Sentiment"):
        cleaned = clean_tweet(user_input)
        vector = tfidf.transform([cleaned]).toarray()
        result = model.predict(vector)[0]

        if result == 4:
            st.success("Sentiment: Positive 😊")
        else:
            st.error("Sentiment: Negative 😔")
