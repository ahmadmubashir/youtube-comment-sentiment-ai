import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = "OPENAI API KEY"
YOUTUBE_API_KEY = "your youtube key"

# Sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

st.title("📊 YouTube Comment Sentiment Analyzer + AI Insights")

video_url = st.text_input("Enter YouTube Video URL")


# Extract video ID
def get_video_id(url):
    video_id = re.search(r"v=([^&]+)", url)
    return video_id.group(1) if video_id else None


# Fetch comments
def get_comments(video_id):
    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        discoveryServiceUrl="https://www.googleapis.com/discovery/v1/apis/{api}/{apiVersion}/rest",
        cache_discovery=False
    )

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults= 1000,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments


if st.button("Analyze Comments"):

    video_id = get_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")

    else:

        st.write("Fetching comments...")

        comments = get_comments(video_id)

        st.write(f"Total comments fetched: {len(comments)}")

        sentiments = []

        for c in comments:
            result = sentiment_model(c[:512])[0]["label"]
            sentiments.append(result)

        df = pd.DataFrame({
            "comment": comments,
            "sentiment": sentiments
        })

        positive = df[df.sentiment == "LABEL_2"].shape[0]
        neutral = df[df.sentiment == "LABEL_1"].shape[0]
        negative = df[df.sentiment == "LABEL_0"].shape[0]

        st.subheader("Sentiment Summary")

        st.write("Positive comments:", positive)
        st.write("Neutral comments:", neutral)
        st.write("Negative comments:", negative)

        # Bar Chart
        fig, ax = plt.subplots()

        ax.bar(
            ["Positive", "Neutral", "Negative"],
            [positive, neutral, negative]
        )

        ax.set_title("YouTube Comment Sentiment")

        st.pyplot(fig)

        # Pie Chart
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [positive, neutral, negative]

        fig2, ax2 = plt.subplots()

        ax2.pie(sizes, labels=labels, autopct='%1.1f%%')

        ax2.set_title("Comment Sentiment Distribution")

        st.pyplot(fig2)

        st.subheader("Sample Comments")

        st.dataframe(df.head(20))

        # LangChain AI Summary
        st.subheader("AI Summary of Viewer Opinions")

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=OPENAI_API_KEY
        )

        prompt = PromptTemplate.from_template(
        """
        Analyze the overall audience opinion from these YouTube comments:

        {comments}

        Provide a short summary explaining what viewers liked or disliked.
        """
        )

        chain = prompt | llm | StrOutputParser()

        summary = chain.invoke({
            "comments": "\n".join(comments[:30])
        })

        st.write(summary)

        # Download comments
        st.download_button(
            "Download comments CSV",
            df.to_csv(index=False),
            "youtube_comments.csv"
        )
