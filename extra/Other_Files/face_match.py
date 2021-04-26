import face_recognition

image_of_erna = face_recognition.load_image_file('known_people/Erna Solberg.jpg')
# face encodings will give us the facial features we can compare to other images
# the command outputs an array, but we just want the first item
erna_face_encoding = face_recognition.face_encodings(image_of_erna)[0]
print(erna_face_encoding)

# comparing with another photo of erna
unknown_image = face_recognition.load_image_file('unknown_people/erna2.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
print(unknown_face_encoding)

# comparing
results = face_recognition.compare_faces([erna_face_encoding], unknown_face_encoding)

# compare_faces will output a boolean, Checking for trur/false
if results[0]:
    print('The unknown encoding is Erna')
else:
    print('The unknown encoding is not Erna')
