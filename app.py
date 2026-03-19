import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
st.set_page_config(page_title="Text Summarization with Langchain", layout="wide")
st.title("Text Summarization with Langchain")

with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=groq_api_key
)

prompt = PromptTemplate(
    template="You are a helpful assistant that summarizes the text:\n{text}",
    input_variables=["text"]
)

url = st.text_input("URL")

if st.button("Summarize"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter all fields")

    elif not validators.url(url):
        st.error("Invalid URL")

    else:
        try:
            with st.spinner("Loading..."):

                # ✅ MOVE HERE (IMPORTANT)
                llm = ChatGroq(
                    model="llama3-70b-8192",
                    groq_api_key=groq_api_key
                )

                # if "youtube.com" in url:
                #     from youtube_transcript_api import YouTubeTranscriptApi

                #     video_id = url.split("v=")[-1]

                #     transcript = YouTubeTranscriptApi.get_transcript(video_id)

                #     text = " ".join([i["text"] for i in transcript])

                #     from langchain_core.documents import Document
                #     data = [Document(page_content=text)]
                # else:
                #     loader = UnstructuredURLLoader(urls=[url])
                
                if "youtube.com" in url:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    from langchain_core.documents import Document

                    video_id = url.split("v=")[-1]

                    transcript = YouTubeTranscriptApi.get_transcript(video_id)

                    text = " ".join([i["text"] for i in transcript])

                    data = [Document(page_content=text)]

                else:
                    loader = UnstructuredURLLoader(urls=[url])
                    data = loader.load()   # ✅ yaha hona chahiye

                # data = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output = chain.run(data)

                st.success(output)

        except Exception as e:
            st.error(f"Exception: {e}")