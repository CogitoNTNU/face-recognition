import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform


# More complicated
# This example is using multiprocess.


# Get next worker's id
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Get previous worker's id
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()


# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[:, :, ::-1]

        # Find all the faces and face encodings in the frame of video, cost most time
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)


if __name__ == '__main__':

    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    if cpu_count() > 2:
        worker_num = cpu_count() - 1  # 1 for capturing frames
    else:
        worker_num = 2

    # Subprocess list
    p = []

    # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

    # Loading pictures in
    karoline_image1 = face_recognition.load_image_file("images/karoline/karoline1.jpg")
    karoline_image2 = face_recognition.load_image_file("images/karoline/karoline2.jpg")
    karoline_image3 = face_recognition.load_image_file("images/karoline/karoline3.jpg")
    karoline_image4 = face_recognition.load_image_file("images/karoline/karoline4.jpg")
    karoline_image5 = face_recognition.load_image_file("images/karoline/karoline5.jpg")
    karoline_image6 = face_recognition.load_image_file("images/karoline/karoline6.jpg")

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
    Global.known_face_encodings = [
        karoline_image1_encoding, karoline_image2_encoding,
        karoline_image3_encoding, karoline_image4_encoding,
        karoline_image5_encoding, karoline_image6_encoding,
        preben_image1_encoding, preben_image2_encoding,
        preben_image3_encoding, preben_image4_encoding,
        preben_image5_encoding, preben_image6_encoding,
        jose_image1_encoding, jose_image2_encoding,
        jose_image3_encoding, jose_image4_encoding,
        jose_image5_encoding
    ]
    Global.known_face_names = [
        "Karoline", "Karoline", "Karoline", "Karoline",
        "Karoline", "Karoline", "Preben", "Preben"
        "Preben", "Preben", "Preben", "Preben",
        "Jose", "Jose", "Jose", "Jose", "Jose"
    ]

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / numpy.sum(fps_list)
            print("fps: %.2f" % fps)

            # Calculating frame delay
            # When fps is higher, the ratio should be smaller

            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()