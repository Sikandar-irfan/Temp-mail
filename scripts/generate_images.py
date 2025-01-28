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

    # Save the logo
    os.makedirs('assets', exist_ok=True)
    image.save('assets/logo.png')

def create_demo_screenshot():
    # Create a terminal-like screenshot
    width = 800
    height = 600
    image = Image.new('RGB', (width, height), '#282C34')
    draw = ImageDraw.Draw(image)

    # Draw menu items
    menu_items = [
        "1. Generate new email",
        "2. Monitor emails",
        "3. List active emails",
        "4. Check messages",
        "5. Forward email",
        "6. Export emails",
        "7. Delete email",
        "8. Clear screen",
        "9. Exit"
    ]

    y = 50
    for item in menu_items:
        draw.text((50, y), item, fill='white')
        y += 40

    # Save the demo screenshot
    os.makedirs('docs/images', exist_ok=True)
    image.save('docs/images/demo.png')

def create_menu_screenshot():
    # Create a menu screenshot
    width = 600
    height = 400
    image = Image.new('RGB', (width, height), '#282C34')
    draw = ImageDraw.Draw(image)

    # Draw menu title
    draw.text((50, 50), "TempMail Manager Menu", fill='#61AFEF')

    # Draw menu items
    menu_items = [
        "📧 Generate Email",
        "👀 Monitor Inbox",
        "📝 List Emails",
        "📬 Check Messages",
        "↗️ Forward Email"
    ]

    y = 100
    for item in menu_items:
        draw.text((50, y), item, fill='white')
        y += 40

    # Save the menu screenshot
    image.save('docs/images/menu.png')

if __name__ == "__main__":
    create_logo()
    create_demo_screenshot()
    create_menu_screenshot()
