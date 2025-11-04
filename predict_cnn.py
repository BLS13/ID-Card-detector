import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("cnn_model.h5")

# Class labels (ensure your folders were 'present' and 'missing')
labels = ["missing", "present"]

def predict_image(frame):
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "present" if pred > 0.5 else "missing"

# Try laptop front camera first (index 1)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera 1 not available. Trying default (0)...")
    cap = cv2.VideoCapture(0)

print("Press 'c' to capture & classify, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed - Press 'c' to Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        prediction = predict_image(frame)
        label_text = f"✅ {prediction.upper()}" if prediction == "present" else f"❌ {prediction.upper()}"
        color = (0, 255, 0) if prediction == "present" else (0, 0, 255)
        cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Prediction", frame)
        cv2.waitKey(1500)  # pause 1.5 sec to show result
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
