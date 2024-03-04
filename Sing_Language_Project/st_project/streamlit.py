import streamlit as st
import cv2

# Open the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Display the live camera feed in Streamlit
while True:
    ret, frame = cap.read()

    if not ret:
        st.warning("Error capturing the camera feed.")
        break

    # Display the frame in Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

# Release the camera when Streamlit app stops
cap.release()
