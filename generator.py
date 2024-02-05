import streamlit as st
from PyPDF2 import PdfReader
import os
from io import BytesIO
from spire.pdf.common import *
from spire.pdf import *
from fastapi.responses import JSONResponse
import openai
from pydantic import BaseModel
from PyPDF2 import PdfReader
import os, re, json
import requests, uuid,concurrent
import nltk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from serpapi import GoogleSearch
from io import BytesIO
from PIL import Image
from moviepy.editor import ImageClip,VideoFileClip, concatenate_videoclips, CompositeVideoClip
import moviepy.editor as mpe
import math
from easyocr import Reader
import tempfile
import os
import base64
def convert_video(input_file, output_format):
    # Load the video clip
    video_clip = VideoFileClip(input_file)

    # Define the output file name based on the input file name and output format
    output_file = f"{input_file.split('.')[0]}_converted.{output_format.lower()}"

    # Define the codec based on the output format
    if output_format == 'MOV':
        codec = 'libx264'  # H.264 codec for MOV
    elif output_format == 'WMV':
        codec = 'wmv2'  # WMV2 codec for WMV
    elif output_format == 'AVI':
        codec = 'libxvid'  # Xvid codec for AVI

    # Export the video clip to the specified format
    video_clip.write_videofile(output_file, codec=codec)

    print(f"Conversion successful. Video saved as {output_file}")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    extracted_text = ''
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text


def summarize_text(text):
    extracted_text = extract_text_from_pdf(pdf_path)
    keyapi = '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'
    max_tokens = 4096  
    truncated_text = extracted_text[:max_tokens]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": truncated_text}
        ],
        max_tokens=300,
        api_key=keyapi
    )
    summary = response['choices'][0]['message']['content']            
    return summary

def paragraph_to_sentences(paragraph):
    cleaned_paragraph = re.sub(r'\n', ' ', paragraph)
    sentences = nltk.sent_tokenize(cleaned_paragraph)
    print(cleaned_paragraph)
    return sentences


def synthesize_and_save(text, gender,voice_name, language_code, output_filename):
  SPEECH_REGION = "eastus"
  SPEECH_KEY = "21cc17a1914042f8b8287972ec739bdc"
  url = f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
  headers = {
      "Ocp-Apim-Subscription-Key": SPEECH_KEY,
      "Content-Type": "application/ssml+xml",
      "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
      "User-Agent": "curl"
  }
  ssml_payload = f'''
  <speak version='1.0' xml:lang='en-US'>
      <voice xml:lang='{language_code}' xml:gender='{gender}' name='{voice_name}'>
        {text}
      </voice>
  </speak>
  '''
  response = requests.post(url, headers=headers, data=ssml_payload.encode('utf-8'))
  if response.status_code == 200:
      with open(output_filename, "wb") as f:
          f.write(response.content)
      print("Text-to-speech conversion successful. Output saved to output.mp3")
  else:
      print(f"Error: {response.status_code}, {response.text}")


def remove_duplicate_images_from_folder(project_id, threshold=0.95):
    hash_list = []
    deduplicated_images = []
    hash_size = 8

    def dhash(image, hash_size=8):
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def hamming_distance(hash1, hash2):
        return bin(hash1 ^ hash2).count('1')

    for filename in os.listdir(f"projects/{project_id}/images"):
        file_path = os.path.join(f"projects/{project_id}/images", filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_hash = dhash(gray_image)

            duplicate_indices = [i for i, h in enumerate(hash_list) if hamming_distance(image_hash, h) <= hash_size * 0.15]

            if not duplicate_indices:
                deduplicated_images.append(image)
                hash_list.append(image_hash)
            else:
                print(f"Removing duplicate: {file_path}")
                # Optionally, you can remove the print statement above if you don't want to display which images are being removed.
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    return {"status":"Duplicated removed"}


def is_folder_empty(folder_path):
    return not any(os.listdir(folder_path))

def scrapeapi(search_term):
    params = {
        "api_key": "a343218c95b0fb09214f5c48baeb0129856aad7680d77387e237984955f33638",
        "engine": "google_images",
        "google_domain": "google.co.in",
        "q": search_term,
        "hl": "hi",
        "gl": "in",
        "location": "Delhi, India"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    image_url = []
    for i in results['images_results']:
        image_url.append(i['original'])
    return image_url


def save_image_from_url(url, save_path, index):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img_name = f"B-{index}.jpg"
        img.save(os.path.join(save_path, img_name))
        print(f"Image saved successfully at {os.path.join(save_path, img_name)}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL {url}: {e}")
    except Exception as e:
        print(f"Error: {e}")



def save_images_concurrently(urls, project_id, max_workers=5):
    save_path = f"projects/{project_id}/images/"
    os.makedirs(save_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(save_image_from_url, url, save_path, i): url for i, url in enumerate(urls)}

        for future in futures:
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing URL {url}: {e}")


def create_video_clip(background, audio_clip, output_filename):
    background = background.set_duration(audio_clip.duration)
    video_clip = background.set_audio(audio_clip)
    video_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=24)
    return video_clip

def create_video(background_video, audio_clips, output_folder):
    background = mpe.VideoFileClip(background_video, audio=False) 
    video_clips = []
    for audio_clip in audio_clips: 
        audio_filename = os.path.splitext(os.path.basename(audio_clip.filename))[0]
        output_filename = f"{output_folder}/{audio_filename}.mp4"
        video_clips.append(create_video_clip(background, audio_clip, output_filename))
    return video_clips


def read_audio_clips(audio_folder):
    audio_clips = []
    for filename in os.listdir(audio_folder):
        if filename.endswith(('.mp3', '.wav')):  # Add more audio formats if needed
            audio_path = os.path.join(audio_folder, filename)
            audio_clip = mpe.AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
    return audio_clips

def audio_clips_duration(audio_folder):
    audio_clips_duration = 0
    num_audio_clips = 0
    for filename in os.listdir(audio_folder):
        if filename.endswith(('.mp3', '.wav')):  
            num_audio_clips += 1
            audio_path = os.path.join(audio_folder, filename)
            audio_clip = mpe.AudioFileClip(audio_path)
            audio_clips_duration += audio_clip.duration
    return audio_clips_duration , num_audio_clips


def load_and_repeat_images(images_folder, num_images_required):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    group_a = [image for image in image_files if image.startswith('A')]
    group_b = [image for image in image_files if image.startswith('B')]

    image_array = []
    
    for i in group_a:
        if len(image_array) == num_images_required:
            return image_array
        else:
            image_array.append(images_folder + i)

    index = 0
    while len(image_array) < num_images_required:
        image_array.append(images_folder+group_b[index % len(group_b)])
        index += 1

    return image_array

def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)

        result = np.array(img)
        img.close()

        return result

    return clip.fl(effect).set_position(('center', 'center'))


def concatenate_video_clips(video_clips, output_file):
    # Load each video clip
    clips = [VideoFileClip(clip_path) for clip_path in video_clips]

    # Concatenate the video clips
    final_clip = concatenate_videoclips(clips, method="compose")

    # Write the concatenated video to the output file
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Close the clips to free up resources
    for clip in clips:
        clip.close()

def sort_key(path):
    return int(path.split('_')[-1].split('.')[0])


def read_video_clips(video_folder):
    video_clips = []
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4','.mov')):  # Add more audio formats if needed
            video_path = os.path.join(video_folder, filename)
            video_clips.append(video_path)
    return video_clips

def overlay_videos(video1_path, video2_path, output_path, position=(0, 0), size=None):
    # Load the two video clips
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    min_duration = min(video1.duration, video2.duration)
    video1 = video1.subclip(0, min_duration)
    video2 = video2.subclip(0, min_duration)

    video1 = video1.set_audio(None)

    if size is not None:
        video1 = video1.resize(size)
    final_video = CompositeVideoClip([video2, video1.set_position(position)])

    final_video.write_videofile(output_path, codec="libx264")

    video1.close()
    video2.close()


def image_filter(image_path, text_confidence_threshold=0.5, blur_threshold=50):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    reader = Reader(['en'])
    results = reader.readtext(image_array)
    confidence_scores = [result[2] for result in results]
    aggregate_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    image_cv2 = cv2.imread(image_path)
    laplacian = cv2.Laplacian(image_cv2, cv2.CV_64F)
    variance = laplacian.var()
    is_blurred = variance < blur_threshold
    is_good = aggregate_confidence < text_confidence_threshold 
    return is_good, aggregate_confidence, is_blurred, variance

def create_video_sentence_list(video_folder, sentence_list):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    sorted_video_files = sorted(video_files, key = sort_key)
    return sorted_video_files

# Define EmailRequest model
class EmailRequest:
    def __init__(self, reciever_mail, title, body):
        self.reciever_mail = reciever_mail
        self.title = title
        self.body = body

# Send email function
def send_email(request_data):
    sender_email = "teamanukriti145@gmail.com"
    receiver_email = request_data.reciever_mail
    subject = request_data.title
    body = request_data.body
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "teamanukriti145@gmail.com"
    smtp_password = "etguatggmtanning"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, receiver_email, message.as_string())

    st.success("Email Sent")

# Streamlit UI for sending email
def send_mail_ui():
    st.title("Send Email")
    reciever_mail = st.text_input("Receiver Email")
    title = st.text_input("Title")
    body = st.text_area("Body")
    if st.button("Send Email"):
        request_data = EmailRequest(reciever_mail, title, body)
        send_email(request_data)

# Translate list function
def translate_list(request_data):
    # Placeholder function, replace with your implementation
    pass

# Streamlit UI for translating list
def translate_list_ui():
    # Placeholder UI, replace with your implementation
    pass

# Generate anchor video function
def generate_anchor_video(gender, script, voice_name):
    # Placeholder function, replace with your implementation
    pass

# Streamlit UI for generating anchor video
def generate_anchor_video_ui():
    # Placeholder UI, replace with your implementation
    pass

def generate_video_from_pdf():
    pass