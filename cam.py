import cv2
import tensorflow as tf
import numpy as np

# Load the PoseNet model
model = tf.saved_model.load('https://tfhub.dev/google/tfjs-model/posenet/mobilenet/float/075/1')

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit the model's input requirements
    input_image = cv2.resize(frame, (257, 257))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # Make predictions
    outputs = model.signatures['serving_default'](input_image)

    # Get keypoints with their confidence scores
    keypoints_with_scores = outputs['output_0'].numpy()

    # Draw keypoints on the frame
    draw_keypoints(frame, keypoints_with_scores, 0.5)

    # Display the frame
    cv2.imshow('Body Part Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
