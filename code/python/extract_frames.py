import cv2
import os

Dataset_path = '/content/drive/MyDrive/dataset5/videos'
for folder in os.listdir(Dataset_path):
    folder_path = os.path.join(Dataset_path, folder)
    if os.path.isdir(folder_path):  # Check if it is a directory
        for file_path in os.listdir(folder_path):
            # Define full input video path
            input_video_path = os.path.join(folder_path, file_path)

            # Get the last folder name
            last_folder_name = os.path.basename(folder_path)
            # Create a subfolder based on the last folder name
            output_directory = os.path.join("/content/drive/MyDrive/dataset5/frames", last_folder_name)
            os.makedirs(output_directory, exist_ok=True)

            # Open the video file
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                print(f"Failed to open video {input_video_path}")
                continue

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Determine fps based on video title
            target_fps = 60 if 'l45' in file_path else 24
            frame_interval = int(round(fps / target_fps))

            # Create a directory for this video
            video_name = os.path.splitext(file_path)[0]  # Extract video name without extension
            video_output_dir = os.path.join(output_directory, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames
                if frame_count % frame_interval != 0:
                    frame_count += 1
                    continue

                # Convert the BGR image to RGB and preprocess
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save the preprocessed frame as an image
                output_path = os.path.join(video_output_dir, f'frame_{frame_count // frame_interval}.jpg')
                cv2.imwrite(output_path, frame_rgb)

                frame_count += 1

            # Release resources
            cap.release()
            print(f"All frames from video '{video_name}' processed and saved successfully.")
    else:
        print(f"Skipping file (not a directory): {folder_path}")
