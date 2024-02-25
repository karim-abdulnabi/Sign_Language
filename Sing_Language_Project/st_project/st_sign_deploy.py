# Install libraries 🏗️
import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: '', 26: 'J', 27: 'Z'}

# Create a Streamlit app
st.title("🤟 ***:blue[Sign Language]*** 🤟")
st.success("**Text-to-Speech Recognition** 🔊")
st.image("https://miro.medium.com/v2/resize:fit:665/1*MLudTwKUYiCYQE0cV7p6aQ.png", width=500)

# Initialize Streamlit session state
if 'recognized_word' not in st.session_state:
    st.session_state.recognized_word = ""

# Create a sidebar for buttons on the left side
st.sidebar.header("Actions 🛠️")

# Create buttons
close_button = st.sidebar.button("Close Application ❌")
# Check if the close button is clicked
if close_button:
    st.sidebar.success("**:blue[thank you for using ASL]🙏**")
    st.sidebar.image("https://menlocoa.org/wp-content/uploads/2023/04/Screen-Shot-2023-04-05-at-10.09.30-AM-900x606.png", width=200)
    st.stop()  # Stop the Streamlit application

# Create a button to refresh the app
refresh_button = st.sidebar.button("Refresh App 🔃")
# Check if the refresh button is clicked
if refresh_button:
    st.rerun()

# Create a button to clear all
clear_button = st.sidebar.button("Clear All 🧹")
# Check if the clear button is clicked
if clear_button:
    st.session_state.recognized_word = ""  # Reset the recognized word

# Create a button to translate
translte_button = st.sidebar.button("Translate 🔄 ")
# Check if the translate button is clicked
if translte_button:
    st.text(f"Recognized Word :{st.session_state.recognized_word}")
    # Add your translation logic here

# Create a button to remove the last character
remove_last_button = st.sidebar.button("Remove Last Character ⬅️ ")
# Check if the remove last button is clicked
if remove_last_button:
    if st.session_state.recognized_word:
        st.session_state.recognized_word = st.session_state.recognized_word[:-1]  # Remove the last character

# Create a button to save the recognized text
save_button = st.sidebar.button("Save Text 📥")
# Check if the "Save Text" button is clicked
if save_button:
    # Add your save_recognized_text logic here
    pass

# Initialize variables
sign_start_time = 0
sign_timeout = 1.25
previous_character = ""



# Initialize Streamlit session state
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Create a checkbox to show/hide the camera
show_camera = st.checkbox("Show Camera")

# Update the session state when the checkbox is toggled
st.session_state.show_camera = show_camera



User
give me the all code 

# Install libraries 🏗️
import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: '', 26: 'J', 27: 'Z'}

# Create a Streamlit app
st.title("🤟 ***:blue[Sign Language]*** 🤟")
st.success("**Text-to-Speech Recognition** 🔊")
st.image("https://miro.medium.com/v2/resize:fit:665/1*MLudTwKUYiCYQE0cV7p6aQ.png", width=500)

# Initialize Streamlit session state
if 'recognized_word' not in st.session_state:
    st.session_state.recognized_word = ""

# Create a sidebar for buttons on the left side
st.sidebar.header("Actions 🛠️")

# Create buttons
close_button = st.sidebar.button("Close Application ❌")
# Check if the close button is clicked
if close_button:
    st.sidebar.success("**:blue[thank you for using ASL]🙏**")
    st.sidebar.image("https://menlocoa.org/wp-content/uploads/2023/04/Screen-Shot-2023-04-05-at-10.09.30-AM-900x606.png", width=200)
    st.stop()  # Stop the Streamlit application

# Create a button to refresh the app
refresh_button = st.sidebar.button("Refresh App 🔃")
# Check if the refresh button is clicked
if refresh_button:
    st.rerun()

# Create a button to clear all
clear_button = st.sidebar.button("Clear All 🧹")
# Check if the clear button is clicked
if clear_button:
    st.session_state.recognized_word = ""  # Reset the recognized word

# Create a button to translate
translte_button = st.sidebar.button("Translate 🔄 ")
# Check if the translate button is clicked
if translte_button:
    st.text(f"Recognized Word :{st.session_state.recognized_word}")
    # Add your translation logic here

# Create a button to remove the last character
remove_last_button = st.sidebar.button("Remove Last Character ⬅️ ")
# Check if the remove last button is clicked
if remove_last_button:
    if st.session_state.recognized_word:
        st.session_state.recognized_word = st.session_state.recognized_word[:-1]  # Remove the last character

# Create a button to save the recognized text
save_button = st.sidebar.button("Save Text 📥")
# Check if the "Save Text" button is clicked
if save_button:
    # Add your save_recognized_text logic here
    pass

# Initialize variables
sign_start_time = 0
sign_timeout = 1.25
previous_character = ""



# Initialize Streamlit session state
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Create a checkbox to show/hide the camera
show_camera = st.checkbox("Show Camera")

# Update the session state when the checkbox is toggled
st.session_state.show_camera = show_camera



# Start the camera capture if the checkbox is checked
if st.session_state.show_camera:
    cap = cv2.VideoCapture( )

    # Main application loop
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
        st.image(frame_bytes, caption='Video Feed', use_column_width=True, channels="BGR")
        st.text(f"Recognized Character: {st.session_state.recognized_word}")

