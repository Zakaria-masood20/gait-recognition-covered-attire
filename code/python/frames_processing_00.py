import cv2
import mediapipe as mp
import pandas as pd
import os
import glob
import numpy as np

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Function to process each image and extract landmarks
def process_image(image_path, class_name, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    landmarks = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([
                round(landmark.x, 5),
                round(landmark.y, 5),
                round(landmark.z, 5),
                round(landmark.visibility, 5)
            ])

        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        annotated_path = os.path.join(output_path, class_name, os.path.basename(image_path))
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        cv2.imwrite(annotated_path, annotated_image)

    return landmarks

def process_directory(input_path, output_path):
    all_data = []

    for class_dir in os.listdir(input_path):
        class_path = os.path.join(input_path, class_dir)
        if os.path.isdir(class_path):
            for image_path in glob.glob(class_path + '/*/*.jpg'):
                print(f"Processing {image_path}")
                landmarks = process_image(image_path, class_dir, output_path)
                if landmarks:
                    all_data.append([class_dir] + landmarks)

    if all_data:
        # Calculate columns based on actual data
        num_features = len(all_data[0]) - 1  # Subtracting the class column
        num_landmarks = num_features // 4

        columns = ['class'] + [f'{coord}_{i}' for i in range(num_landmarks) for coord in ['x', 'y', 'z', 'visibility']]

        df = pd.DataFrame(all_data, columns=columns)

        # Ensure all numeric columns are float type
        for column in columns[1:]:  # Skip the 'class' column
            df[column] = pd.to_numeric(df[column], errors='coerce')

        csv_path = os.path.join(output_path, 'gait_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

input_path = '/content/drive/MyDrive/Dataset2/frames_00'
output_path = '/content/drive/MyDrive/Dataset2/frames_00_processed'

process_directory(input_path, output_path)
