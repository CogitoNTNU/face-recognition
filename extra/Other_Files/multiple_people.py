import face_recognition

image = face_recognition.load_image_file('multiple_people/4.jpg')
image2 = face_recognition.load_image_file('multiple_people/7.jpg')

face_locations = face_recognition.face_locations(image)
face_locations2 = face_recognition.face_locations(image2)

# Array of coordinates for each face
print(face_locations)
print(f'there are {len(face_locations)} people in the picture')

print(face_locations2)
print(f'there are {len(face_locations2)} people in the picture')
