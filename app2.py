import cv2
import pygame
from pygame.locals import *
from ultralytics import YOLO
import numpy as np
from pytube import YouTube

# Initialize Pygame
pygame.init()

# Load YOLO model
model = YOLO('yolov8n.pt')

# Get the video capture object (assuming webcam, change the index as needed)
 
youtube_url = "https://www.youtube.com/watch?v=iJZcjZD0fw0&ab_channel=IncredibleIndianTraffic"

# Create a YouTube object
yt = YouTube(youtube_url)

# Get the highest resolution video stream
video_stream = yt.streams.get_highest_resolution()
cap = cv2.VideoCapture(video_stream.url)

# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set Pygame screen dimensions to match frame dimensions
screen = pygame.display.set_mode((frame_width, frame_height))

# Define colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Function to draw text on screen
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)

# Define font for text rendering
font = pygame.font.Font(None, 36)

# Initialize selected object index and its color
current_idx = 0
key_pressed = False
# Initialize overlay once before the main loop
overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

running = True
highlighted_rect = None  # To store the highlighted rectangle coordinates
highlighted_rect_high = None

while running:
    ret, frame = cap.read()  # Read a frame from the video capture

    if not ret:
        break  # Break the loop if no frame is captured

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
    frame_rgb = cv2.flip(frame_rgb, 1)
    
    # Detect objects in the frame using YOLO
    results = model.track(frame_rgb)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                current_idx = max(0, current_idx - 1)
                # Update highlighted_rect position based on the current_idx
                if results and len(results[0].boxes.data) > current_idx:
                    x1, y1, x2, y2, *_ = results[0].boxes.data[current_idx]
                    highlighted_rect = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            elif event.key == pygame.K_RIGHT:
                current_idx = min(len(results[0].boxes.data) - 1, current_idx + 1)
                # Update highlighted_rect position based on the current_idx
                if results and len(results[0].boxes.data) > current_idx:
                    x1, y1, x2, y2, *_ = results[0].boxes.data[current_idx]
                    highlighted_rect = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            elif event.key == pygame.K_RETURN:
                if results and len(results[0].boxes.data) > current_idx:
                    x1, y1, x2, y2, *_ = results[0].boxes.data[current_idx]
                    highlighted_rect = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    highlighted_rect_high = highlighted_rect

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the frame on the screen
    if highlighted_rect_high  and current_idx < len(results[0].boxes.data):
        if len(results[0].boxes.data[current_idx]) >= 5:
            x1, y1, x2, y2, *_ = results[0].boxes.data[current_idx]
            highlighted_rect_high = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            x, y, w, h = highlighted_rect_high
            overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), [255, 0, 0], -1) 
            frame_rgb = cv2.addWeighted(frame_rgb, 1, overlay, 0.6, 0)
    pygame_img = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
    screen.blit(pygame_img, (0, 0))

    # Draw bounding boxes and text on the screen using OpenCV
    if results and len(results[0].boxes.data) != 0:
        for idx, boxes_info in enumerate(results[0].boxes.data):
            if len(boxes_info) >= 7:  # Ensure enough values are available for unpacking
                x1, y1, x2, y2, ob_id, confidence, class_idx = boxes_info
                class_label = model.names[int(class_idx)]
                color = RED if idx == current_idx else GREEN

                # Draw bounding box using pygame.draw.rect
                pygame.draw.rect(screen, color, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 2)
                draw_text(f'{class_label}: {confidence:.2f} : id:{int(ob_id)}', font, color, screen, int(x1), int(y1) - 30)

    pygame.display.update()

pygame.quit()  # Quit Pygame
cap.release()  # Release the video capture object
