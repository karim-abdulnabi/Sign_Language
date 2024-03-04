import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Define a VideoTransformer class to process video frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def transform(self, frame):
        # Read a frame from the webcam
        ret, img = self.cap.read()

        # Perform any image processing if needed
        # For example, you can convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Return the processed frame
        return gray

# Streamlit app
def main():
    st.title("Webcam Video Recorder")

    # Use webrtc_streamer to display the webcam feed and processed video
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        # Display the processed video
        st.image(webrtc_ctx.video_transformer.get_frame())

if __name__ == "__main__":
    main()
