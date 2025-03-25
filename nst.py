import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image
import os  # Added for directory handling

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path)
    img = img.resize(target_size)  # Resize the image
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize and ensure float32
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Paths to content and style images
content_path = r"C:\Users\chari\Desktop\Codetech\NeuralStyleTransfer\content.jpeg"
style_path = r"C:\Users\chari\Desktop\Codetech\NeuralStyleTransfer\style.jpeg"

# Load and preprocess images
content_image = load_and_preprocess_image(content_path)
style_image = load_and_preprocess_image(style_path)

# Convert explicitly to float32 tensor before passing to model
content_image = tf.constant(content_image, dtype=tf.float32)
style_image = tf.constant(style_image, dtype=tf.float32)

# Load pre-trained model from TensorFlow Hub
style_transfer_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Apply style transfer
stylized_image = style_transfer_model(content_image, style_image)[0]

# Convert output image back to uint8
stylized_image = np.array(stylized_image[0] * 255, dtype=np.uint8)

# Ensure the output directory exists
output_dir = r"C:\Users\chari\Desktop\Codetech\NeuralStyleTransfer\output"
os.makedirs(output_dir, exist_ok=True)

# Save output image
output_path = os.path.join(output_dir, "styled_output.jpg")
Image.fromarray(stylized_image).save(output_path)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(stylized_image)
plt.axis("off")
plt.show()

print(f"Styled image saved at: {output_path}")
