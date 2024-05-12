import cv2
import pygame
from pygame.locals import *
from ultralytics import YOLO
import numpy as np

# Initialize Pygame
pygame.init()

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load image using OpenCV (replace 'image_path.jpg' with your image file path)
image_path = 'objects.jpg'
image = cv2.imread(image_path)
original_image = image.copy()  # Make a copy of the original image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

# Get image dimensions
image_height, image_width, _ = image.shape

# Set Pygame screen dimensions to match image dimensions
screen = pygame.display.set_mode((image_width, image_height))

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
results = model.track(source=image)
# Initialize selected object index and its color
current_idx = 0

running = True
highlighted_rect = None  # To store the highlighted rectangle coordinates
overlay = np.zeros_like(image, dtype=np.uint8)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                current_idx = max(0, current_idx - 1)
            elif event.key == pygame.K_RIGHT:
                current_idx = min(len(results[0].boxes.data) - 1, current_idx + 1)
            elif event.key == pygame.K_RETURN:
                # Get the bounding box coordinates of the selected object
                x1, y1, x2, y2, *_ = results[0].boxes.data[current_idx]
                highlighted_rect = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the image on the screen
    pygame_img = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
    screen.blit(pygame_img, (0, 0))

    # Draw bounding boxes and text on the screen using OpenCV
    for idx, boxes_info in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, ob_id, confidence, class_idx = boxes_info
        class_label = model.names[int(class_idx)]
        color = RED if idx == current_idx else GREEN

        # Draw bounding box using pygame.draw.rect
        pygame.draw.rect(screen, color, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 2)
        draw_text(f'{class_label}: {confidence:.2f} : id:{int(ob_id)}', font, color, screen, int(x1), int(y1) - 30)

    # Highlight the area inside the selected bounding box
    if highlighted_rect:
        x, y, w, h = highlighted_rect
        overlay = np.zeros_like(image, dtype=np.uint8)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), [255, 0, 0], -1)  # Fill rectangle with red

    # Apply overlay to the image
    image = cv2.addWeighted(original_image, 1, overlay, 0.6, 0)

    pygame.display.update()

pygame.quit()  # Quit Pygame
