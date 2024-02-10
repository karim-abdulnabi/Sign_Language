#Install libraries 🏗️
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import streamlit as st
import io
from PIL import Image
from translate import Translator
import time
import speech_recognition as sr
from PIL import Image
import requests
import youtube_dl
import re

#pip install opencv-python


sign_start_time = 0
sign_timeout = 1.25

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: '', 26: 'J', 27: 'Z'}

previous_character = ""
recognized_word = "" # Variable to store the recognized word



# Create a Streamlit app

# Add a title to your Streamlit app
st.title("🤟 ***:blue[Sign Language]*** 🤟")

st.success("**Text-to-Speech Recognition** 🔊")


# Display the image containing sign language signs
st.image("https://miro.medium.com/v2/resize:fit:665/1*MLudTwKUYiCYQE0cV7p6aQ.png", width=500 )

# The rest of your Streamlit code...
  

# Create a text element to display the recognized character
recognized_text = st.empty()

# Initialize Streamlit session state
if 'recognized_word' not in st.session_state:
    st.session_state.recognized_word = "" 

# Initialize the TTS engine
def initialize_tts_engine():
    if not hasattr(st.session_state, "tts_engine"):
        st.session_state.tts_engine = pyttsx3.init()

# Initialize the TTS engine at the beginning of the app
initialize_tts_engine()

# Display the video feed and recognized character
video_frame = st.empty()

# Create a sidebar for buttons on the left side
st.sidebar.header("Actions 🛠️")

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Create buttons
close_button = st.sidebar.button("Close Application ❌")
# Check if the close button is clicked
if close_button:
    st.sidebar.success("**:blue[thank you for using ASL]🙏**")
    st.sidebar.image("https://menlocoa.org/wp-content/uploads/2023/04/Screen-Shot-2023-04-05-at-10.09.30-AM-900x606.png", width=200 )
    st.stop()  # Stop the Streamlit application


# Function to reset the app state
def reset_app_state():
    st.session_state.recognized_word = ""
    st.session_state.translated_word = ""
    # Add a button to refresh the app

refresh_button = st.sidebar.button("Refresh App 🔃")

# Check if the refresh button is clicked
if refresh_button:
    st.rerun()

# Create a button to clear all
clear_button = st.sidebar.button("Clear All 🧹")

# Check if the clear button is clicked
if clear_button:
    st.session_state.recognized_word = ""  # Reset the recognized word

# Create a button to speak the recognized text
speak_button = st.sidebar.button("Speak Recognized Text 🔊")

# Check if the speak button is clicked
if speak_button:
    engine.say(st.session_state.recognized_word)
    engine.runAndWait()

translte_button = st.sidebar.button("Translate 🔄 ")
# Check if the show word button is clicked
if translte_button:
    st.text(f"Recognized Word :{st.session_state.recognized_word}")
    translte = Translator(to_lang='ar')
    st.success(f"**translate to arabic :** {translte.translate(st.session_state.recognized_word)}")
   


# Create a button to remove the last character
remove_last_button = st.sidebar.button("Remove Last Character ⬅️ ")

# Check if the remove last button is clicked
if remove_last_button:
    if st.session_state.recognized_word:
        st.session_state.recognized_word = st.session_state.recognized_word[:-1]  # Remove the last character

# Create a button to save the recognized text
save_button = st.sidebar.button("Save Text 📥")

#####

# Add a button for speech recognition
speech_recognition_button = st.sidebar.button("Start Speech Recognition 🗣️")

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to perform speech recognition
def perform_speech_recognition():
    with sr.Microphone() as source:
        st.sidebar.write("Speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        recognized_speech = recognizer.recognize_google(audio)  # You can choose a different recognizer if needed
        st.sidebar.write(f"Recognized Speech: {recognized_speech}")
        st.session_state.recognized_word += recognized_speech  # Append the recognized speech to the text
    
    except sr.UnknownValueError:
        st.sidebar.write("Could not understand the audio")
    except sr.RequestError as e:
        st.sidebar.write(f"Error: {e}")

# Check if the speech recognition button is clicked
if speech_recognition_button:
    perform_speech_recognition()


#####
# Initialize a variable to store the saved text
recognaized_text = ""

# Function to save the recognized text to a file
def save_recognized_text():
    global recognaized_text
    recognaized_text = st.session_state.recognized_word
    
    # Specify the file path where you want to save the text
    file_path = "recognaized_text.txt"

    # Save the text to the file
    with open(file_path, "w") as file:
        file.write(recognaized_text)

# Check if the "Save Text" button is clicked
if save_button:
    save_recognized_text()


# Display a success message if text is saved
if recognaized_text:
    st.success(f"Text saved: {recognaized_text}")
    st.write(f"The text has been saved to a file: recognaized_text.txt")



cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW)  # Start the camera capture

while True:
    ret, frame = cap.read()

    if not ret:
        # Video capture failed, wait and try again
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        if predicted_character == previous_character:
            if sign_start_time is None:
                sign_start_time = time.time()
            else:
                current_time = time.time()
                if current_time - sign_start_time >= sign_timeout:
                    st.session_state.recognized_word += predicted_character
                    sign_start_time = None
        else:
            sign_start_time = None

        previous_character = predicted_character

    # Convert the frame to bytes
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

    # Update the video feed and recognized character using Streamlit
    video_frame.image(frame_bytes, caption='Video Feed', use_column_width=True, channels="BGR")
    recognized_text.text(f"Recognized Character: {st.session_state.recognized_word}")
    
    

# Close the video capture and the app
cap.release()
cv2.destroyAllWindows()
