import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Perform any image processing here (optional)
        # In this example, we simply return the input frame
        return frame

# Streamlit app
def main():
    st.title("WebRTC Video Stream Example")

    # Create a WebRTC streaming context
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        mode="sendrecv",  # WebRTCMode.SENDRECV
        async_processing=True,
    )

    if webrtc_ctx.video_transformer:
        # Display the video feed
        st.image(webrtc_ctx.image_out, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
