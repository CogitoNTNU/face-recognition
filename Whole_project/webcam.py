import face_recognition
import cv2
import numpy as np

# Easier
# Works by:
#   1. Processing each video frame at 1/4 resolution (though still displays it in full resolution)
#   2. Only detecting faces in every other frame of video.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Loading pictures in
karoline_image1 = face_recognition.load_image_file("images/karoline1.jpg")
karoline_image2 = face_recognition.load_image_file("images/karoline2.jpg")
karoline_image3 = face_recognition.load_image_file("images/karoline3.jpg")
karoline_image4 = face_recognition.load_image_file("images/karoline4.jpg")
karoline_image5 = face_recognition.load_image_file("images/karoline5.jpg")
karoline_image6 = face_recognition.load_image_file("images/karoline6.jpg")

# Learn how to recognize the photos
karoline_image1_encoding = face_recognition.face_encodings(karoline_image1)[0]
karoline_image2_encoding = face_recognition.face_encodings(karoline_image2)[0]
karoline_image3_encoding = face_recognition.face_encodings(karoline_image3)[0]
karoline_image4_encoding = face_recognition.face_encodings(karoline_image4)[0]
karoline_image5_encoding = face_recognition.face_encodings(karoline_image5)[0]
karoline_image6_encoding = face_recognition.face_encodings(karoline_image6)[0]

# Loading pictures in
preben_image1 = face_recognition.load_image_file("images/preben/preben1.jpg")
preben_image2 = face_recognition.load_image_file("images/preben/preben2.jpg")
preben_image3 = face_recognition.load_image_file("images/preben/preben3.jpg")
preben_image4 = face_recognition.load_image_file("images/preben/preben4.jpg")
preben_image5 = face_recognition.load_image_file("images/preben/preben5.jpg")
preben_image6 = face_recognition.load_image_file("images/preben/preben6.jpg")

# Learn how to recognize the photos
preben_image1_encoding = face_recognition.face_encodings(preben_image1)[0]
preben_image2_encoding = face_recognition.face_encodings(preben_image2)[0]
preben_image3_encoding = face_recognition.face_encodings(preben_image3)[0]
preben_image4_encoding = face_recognition.face_encodings(preben_image4)[0]
preben_image5_encoding = face_recognition.face_encodings(preben_image5)[0]
preben_image6_encoding = face_recognition.face_encodings(preben_image6)[0]

# Loading pictures in
jose_image1 = face_recognition.load_image_file("images/jose/jose1.PNG")
jose_image2 = face_recognition.load_image_file("images/jose/jose2.JPG")
jose_image3 = face_recognition.load_image_file("images/jose/jose3.JPG")
jose_image4 = face_recognition.load_image_file("images/jose/jose4.JPG")
jose_image5 = face_recognition.load_image_file("images/jose/jose5.JPG")

# Learn how to recognize the photos
jose_image1_encoding = face_recognition.face_encodings(jose_image1)[0]
jose_image2_encoding = face_recognition.face_encodings(jose_image2)[0]
jose_image3_encoding = face_recognition.face_encodings(jose_image3)[0]
jose_image4_encoding = face_recognition.face_encodings(jose_image4)[0]
jose_image5_encoding = face_recognition.face_encodings(jose_image5)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    karoline_image1_encoding, karoline_image2_encoding,
    karoline_image3_encoding, karoline_image4_encoding,
    karoline_image5_encoding, karoline_image6_encoding,
    preben_image1_encoding, preben_image2_encoding,
    preben_image3_encoding, preben_image4_encoding,
    preben_image5_encoding, preben_image6_encoding,
    jose_image1_encoding, jose_image2_encoding,
    jose_image3_encoding, jose_image4_encoding,
    jose_image5_encoding ]

known_face_names = [
    "Karoline", "Karoline", "Karoline", "Karoline",
    "Karoline", "Karoline", "Preben", "Preben"
    "Preben", "Preben", "Preben", "Preben",
    "Jose", "Jose", "Jose", "Jose", "Jose"
]

# Initializing variables to use later
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size (faster face recognition processing)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Finding the known face that matches with the smallest distance (within threshold)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()