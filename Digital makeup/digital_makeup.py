from PIL import Image, ImageDraw
import face_recognition

# Loading picture (JPG) to numpy array
image = face_recognition.load_image_file("known_people/AudunLysbakken.jpg")

# Using face_landmarks to find all the facial features in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Converting from array to PIL so that we can draw on face
pil_image = Image.fromarray(image)

# Applying makeup
for face_landmarks in face_landmarks_list:
    draw = ImageDraw.Draw(pil_image, 'RGBA')

    # Eyebrows
    draw.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    draw.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    draw.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    draw.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Lips (lip gloss)
    draw.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    draw.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    draw.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    draw.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # Eyes (sparkle)
    draw.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    draw.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # Eyeliner
    draw.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    draw.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    # Showing image with makeup
    pil_image.show()
