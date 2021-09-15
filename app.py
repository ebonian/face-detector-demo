import cv2

# Configure to use webcam or image
# webcam = 0
# image = 1
method = 0


# Load pre-trained face data
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Load webcam
webcam = cv2.VideoCapture(0)

# Load image
img = cv2.imread('EM2.jpg')

if method == 0:
    # Video Method
    while True:
        successful_frame_read, frame = webcam.read()

        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

        cv2.imshow('Image', frame)
        key = cv2.waitKey(1)

        # Break when press q
        if key == 81 or key == 113:
            break
elif method == 1:
    # Image Method
    # Convert to B&W
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face from pre-trained data and img
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

    # Display img
    cv2.imshow('Image', img)
    cv2.waitKey()

webcam.release()

print("Code Completed")
