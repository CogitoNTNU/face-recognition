# Class that identifies faces in a picture, and compares them to known images and their encodings.
# Draws a box around all faces, with labels associated with names or "Unknown Person"

import face_recognition
from PIL import Image, ImageDraw
# ImageDraw is also imported because we are going to be drawing on top of an image(the box)

image_of_erna = face_recognition.load_image_file('known_people/Erna Solberg.jpg')
# face encodings will give us the facial features we can compare to other images
# the command outputs an array, but we just want the first item
erna_face_encoding = face_recognition.face_encodings(image_of_erna)[0]

image_of_bent = face_recognition.load_image_file('known_people/Bent Høye.jpg')
bent_face_encoding = face_recognition.face_encodings(image_of_bent)[0]

# Create an array of encodings and names
known_face_encodings = [erna_face_encoding, bent_face_encoding]
known_face_names = ["Erna Solberg", "Bent Høye"]

# Loading test image to find faces in:
test_image = face_recognition.load_image_file("known_people/BentHøyeOgSquad.jpg")

# Finding faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# We need to convert the test_image to "pillow-format" (PIL) so that we can draw on it
pil_image = Image.fromarray(test_image)

# Create an ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Now we have to loop through the faces in the test image to find the correct people
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown Person"

    # If we find a match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # First we're going to draw a box around the face
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))  # (0,0,0) == black outline

    # Now we're going to draw a label containing the name (rectangle and text)
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# Deleting the draw instance outside of the for-loop
del draw

# Finally: displaying the PIL image
pil_image.show()

# We can also save image if we want:
pil_image.save('identify.jpg')
