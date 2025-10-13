from PIL import Image

# Create a simple 100x100 black image
img = Image.new('RGB', (100, 100), 'black')
img.save('test_image.png')