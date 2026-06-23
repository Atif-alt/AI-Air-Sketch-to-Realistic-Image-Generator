import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

print("Loading AI Models...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
)

pipe = pipe.to("cpu")

print("AI Ready!")

os.makedirs("sketches", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x = None
prev_y = None

PROMPT = "A realistic modern house"

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            if prev_x is None:
                prev_x, prev_y = x, y

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (x, y),
                (255, 255, 255),
                5
            )

            prev_x, prev_y = x, y

    else:
        prev_x = None
        prev_y = None

    combined = cv2.add(frame, canvas)

    cv2.putText(
        combined,
        "G=Generate  C=Clear  Q=Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("AI Air Sketch Generator", combined)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    elif key == ord('g'):

        cv2.imwrite("sketches/sketch.png", canvas)

        print("Generating AI Image...")

        image = Image.open("sketches/sketch.png").convert("RGB")

        result_img = pipe(
            prompt=PROMPT,
            image=image,
            num_inference_steps=10
        ).images[0]

        result_path = os.path.abspath("outputs/result.png")

        result_img.save(result_path)

        print(f"Saved: {result_path}")

        try:
            os.startfile(result_path)
        except:
            pass

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()