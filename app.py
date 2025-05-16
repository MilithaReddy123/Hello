import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from deepface import DeepFace
from liveness import is_real_face
import asyncio
import gc
import psutil
import sys

# Fix for Windows event loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="⚡ Fast Emotion Detector", layout="wide")
st.title("⚡ Low-Lag Real-time Emotion Detection with Anti-Spoofing")

# RTC config for WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Cache the emotion model
@st.cache_resource
def load_emotion_model():
    return DeepFace.build_model('Emotion')

emotion_model = load_emotion_model()

class EmotionLivenessProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.result_emotion = "Initializing..."
        self.result_score = 0.0
        self.live_result = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        display_img = img.copy()

        # Analyze every 30th frame to reduce load
        if self.frame_count % 30 == 0:
            try:
                # Resize to speed up analysis
                small_img = cv2.resize(img, (96, 96))

                # Liveness detection
                self.live_result = is_real_face(small_img)

                if self.live_result:
                    result = DeepFace.analyze(
                        small_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        models={'emotion': emotion_model}
                    )
                    self.result_emotion = result[0]['dominant_emotion']
                    self.result_score = result[0]['emotion'][self.result_emotion]
                    del result
                else:
                    self.result_emotion = "Fake Face"
                    self.result_score = 0.0

                gc.collect()

            except Exception as e:
                print("[ERROR]", e)
                self.result_emotion = "Error"
                self.result_score = 0.0

        label = f"{self.result_emotion} ({self.result_score:.1f}%)"
        label += " | ✅ Real" if self.live_result else " | ❌ Fake"
        color = (0, 255, 0) if self.live_result else (0, 0, 255)

        cv2.putText(display_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# WebRTC Stream
webrtc_streamer(
    key="live-emotion-detector",
    video_processor_factory=EmotionLivenessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# Optional: Memory usage sidebar
process = psutil.Process()
mem = process.memory_info().rss / 1e6  # MB
st.sidebar.markdown(f"### RAM Usage: `{mem:.2f} MB`")
