# Image -> Image caption using BLIP model -> Short story using OpenAI chatgpt -> Audio using OpenAI TTS
# https://www.youtube.com/watch?v=YwzoVmO_ppQ

# Run with streamlit run app.py - Starts a web server (default localhost:8501) and opens the app in your browser. 
# Streamlit: Hosts the UI loop; Re-runs the entire script top to bottom; Provides the runtime that makes st.* calls work

# python app.py - would fail immediately because st.secrets, st.file_uploader, etc. require the Streamlit runtime to exist

# For HTTP requests to external APIs
import requests
import streamlit as st

# HF library to run pre-trained model locally - BLIP image captioning
from transformers import pipeline

# Langchain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_openai import ChatOpenAI

#OPENAI api key from .streamlit/secrets.toml
def get_api_key(key_name):
    return st.secrets[key_name]

# Bootstrapping Language-Image Pre-training (BLIP): vision-language model from Salesforce; describes images in natural language
# pipeline downloads the BLIP model to ~/.cache/huggingface/hub, loads into RAM, CPU/GPU
# reloaded into RAM on each call of img2text()
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # returns a list like: [{'generated_text': 'text'}]; [0] — takes the first result from the list; ['generated_text'] — extracts text
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# converts image caption to short story
def generate_story(scenario):
    template = """ 
    You are a storyteller. You can generate a short story based on a simple narrative. The story should be no more than 200 words;
    CONTEXT: {scenario}
    STORY:

    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=get_api_key('OPENAI_API_KEY'),
        ),
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.run(scenario)
    print(story)
    return story

# Short story to .flac audio file (Free Lossless Audio Codec - compressed but no quality is lost)
# https://platform.openai.com/docs/guides/text-to-speech
def text2speech(message):
    API_URL="https://api.openai.com/v1/audio/speech"
    headers= {"Authorization": f"Bearer {get_api_key('OPENAI_API_KEY')}"}
    payload={"model": "gpt-4o-mini-tts",
    "input": message,
    "format": "flac"}

    response = requests.post(API_URL, headers=headers, json=payload)

    # Add error handling
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        st.error(f"TTS API Error: {response.status_code}")
        return
    
    with open("audio.flac", "wb") as f:
        f.write(response.content)

def main():
    # Streamlit page UI
    st.set_page_config(page_title="Image to Audio Story", page_icon="🎧")
    st.header("Turn image into Audio Story")

    # File uploader
    uploaded_file=st.file_uploader("Choose an image ...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        # Save uploaded file locally
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # Display image
        st.image(uploaded_file, caption='Uploaded Image.', width=400)

        # Image to caption
        scenario = img2text(uploaded_file.name)

        # Caption to short story
        story = generate_story(scenario)

        # Short story to audio
        text2speech(story)

        # Display
        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("audio.flac")

# "only run main() if this file is executed directly, not imported."
if __name__ == "__main__":
    main()