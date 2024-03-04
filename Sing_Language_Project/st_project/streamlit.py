import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Perform any image processing here
        # For example, you can convert the frame to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img

def main():
    st.title("Webcam Video Recorder")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        mode=WebRtcMode.SENDRECV,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        st.image(webrtc_ctx.video_transformer.frame_out, channels="GRAY")

if __name__ == "__main__":
    main()
