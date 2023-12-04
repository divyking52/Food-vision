import streamlit as st
import subprocess
from pytube import YouTube
from pathlib import Path
import os
import whisper
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model():
    model = whisper.load_model("base")
    return model


def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("error occured while downloading")
    print("download completed")

    return video_filename


def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file,file_name)

    audio_filename= Path(file_name).stem+'.mp3'
    video_filename = save_video(url,Path(file_name).stem+'.mp4')
    print(yt.title+ 'has been successfully downloaded')
    return yt.title, audio_filename, video_filename

def audio_to_transcription(audio_filename):
    model = load_model()
    result= model.transcribe(audio_filename)
    transcript = result['text']
    return transcript

def load_gpt2_model():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_recipe_with_gpt2(prompt, model, tokenizer):
    prompt= "write the food recipe from the below text:\n"+ prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=200,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)

    recipe = tokenizer.decode(output[0], skip_special_tokens=True)
    return recipe


# Load GPT-2 model and tokenizer
gpt2_model, gpt2_tokenizer = load_gpt2_model()


def process_video(url):
    video_title, audio_filename, video_filename = save_audio(url)
    st.video(video_filename)

    # Transcription
    model = load_model()
    result = model.transcribe(audio_filename)
    transcript = result['text']

st.set_page_config(layout='wide')
st.subheader('Food Vision')

url = st.text_input("enter the link of your YT video")

if url is not None:
        if st.button('Generate'):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.info('Video uploaded successfully')
                video_title, audio_filename, video_filename = save_audio(url)
                st.video(video_filename)
            with col2:
                st.info('transcript is below')
                print(audio_filename)
                transcript_result = audio_to_transcription(audio_filename)
                st.success(transcript_result)

            with col3:
                st.info("recipe is generated below")
                recipe_result = generate_recipe_with_gpt2(video_title, gpt2_model, gpt2_tokenizer)
                st.success(recipe_result)




    # st.subheader("Transcript")
    # st.success(transcript)

    # # Recipe Generation with GPT-2
    # st.subheader("Generated Recipe")
    # recipe_result = generate_recipe_with_gpt2(transcript, gpt2_model, gpt2_tokenizer)
    # st.success(recipe_result)


