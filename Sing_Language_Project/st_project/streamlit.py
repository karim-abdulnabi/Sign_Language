import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Create a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Display the frame in Streamlit
        st.image(frame.to_ndarray(format="bgr24"), channels="BGR", use_column_width=True)

# Use webrtc_streamer to open the camera and display the live feed
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
