import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_queue = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Perform any image processing here
        # For example, you can convert the frame to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Update the frame queue
        if self.frame_queue is not None:
            self.frame_queue.put(gray_img)
        
        return gray_img

def main():
    st.title("Webcam Video Recorder")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        mode=WebRtcMode.SENDRECV,
        async_transform=True,
    )

    video_transformer = webrtc_ctx.video_transformer

    if video_transformer:
        st.image(video_transformer.frame_out, channels="GRAY")

        if not hasattr(video_transformer, "frame_queue"):
            video_transformer.frame_queue = st.queue(max_queue_size=1)

        frame_queue = video_transformer.frame_queue

        if frame_queue is not None:
            while True:
                if st.stop_button("Stop Recording"):
                    break

                if not frame_queue:
                    continue

                try:
                    frame = frame_queue.get_nowait()
                    st.image(frame, channels="GRAY", use_container_width=True)
                except st.Empty:
                    pass

if __name__ == "__main__":
    main()
