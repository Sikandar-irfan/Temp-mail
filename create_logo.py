from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with a white background
    width = 500
    height = 500
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Draw a blue envelope shape
    envelope_points = [
        (100, 150),  # Top left
        (400, 150),  # Top right
        (400, 350),  # Bottom right
        (100, 350),  # Bottom left
    ]
    draw.polygon(envelope_points, fill='#4A90E2')

    # Draw the envelope flap
    flap_points = [
        (100, 150),  # Left
        (250, 250),  # Middle
        (400, 150),  # Right
    ]
    draw.polygon(flap_points, fill='#357ABD')

    # Draw a temporary symbol (⌛)
    draw.text((200, 200), "⌛", fill='white', font=ImageFont.truetype("DejaVuSans.ttf", 120))

    # Save the image
    os.makedirs('assets', exist_ok=True)
    image.save('assets/logo.png')

if __name__ == "__main__":
    create_logo()
