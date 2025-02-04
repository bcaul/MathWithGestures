import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Create UI layout
col1, col2 = st.columns([3, 2])

with col1:
    # Initialize webcam toggle state
    if "run" not in st.session_state:
        st.session_state.run = True
    st.session_state.run = st.checkbox('Run', value=st.session_state.run)
    
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    if "ai_output" not in st.session_state:
        st.session_state.ai_output = ""
    output_text_area = st.subheader(st.session_state.ai_output)

# Configure the AI model
genai.configure(api_key="what are you looking for exactly? -_-") #Enter your own API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Initialize hand detector
detector = HandDetector(
    staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5
)

# Initialize drawing canvas and previous position
if "canvas" not in st.session_state:
    st.session_state.canvas = None
if "prev_pos" not in st.session_state:
    st.session_state.prev_pos = None

# Function to detect hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

# Function to draw on the canvas based on hand gestures
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger up (drawing mode)
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    
    elif fingers == [1, 1, 1, 1, 1]:  # Open palm (clear screen)
        canvas = np.zeros_like(canvas)

    return current_pos, canvas

# Function to send image to AI
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # If four fingers are up, trigger AI processing
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Guess this drawing.", pil_image])
        return response.text
    return None

# Main Streamlit loop
while st.session_state.run:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        st.session_state.prev_pos, st.session_state.canvas = draw(info, st.session_state.prev_pos, st.session_state.canvas)
        ai_output = sendToAI(model, st.session_state.canvas, fingers)
        if ai_output:
            st.session_state.ai_output = ai_output

    # Update UI elements inside the Streamlit execution flow
    output_text_area.text(st.session_state.ai_output)
    image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
