import cv2
import numpy as np
import requests
from PIL import Image
import io

jetbot_ip = '194.47.156.221'


# Parametrar
KNOWN_WIDTH = 3.5  # iPhone bredd i cm
FOCAL_LENGTH = 500  # Byt till kalibrerat värde!

def get_jetbot_frame(jetbot_ip):
    response = requests.get(f'http://{jetbot_ip}:8080/camera')
    image = Image.open(io.BytesIO(response.content))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame

while True:
    try:
        frame = get_jetbot_frame(jetbot_ip)

        # LAB konvertering
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        _, mask = cv2.threshold(A, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 15:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Avståndsberäkning
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
                cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Visa
        cv2.imshow("Jetbot Red Object + Distance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

cv2.destroyAllWindows()
