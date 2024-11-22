import cv2

# Load the pre-trained cascade classifiers for detecting the face and the nose
face_cascade = cv2.CascadeClassifier(r"C:\Users\rkssp\OneDrive\Desktop\MAIN TELFLOGIC\Human Pose Estimation\haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier(r'C:/Users/rkssp/Downloads/haarcascade_mcs_nose.xml')


# Function to detect face mask using webcam
def detect_face_mask():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Loop through each face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get the region of interest (ROI) containing the nose
            roi_gray = gray[y:y + h, x:x + w]

            # Detect noses in the ROI
            noses = nose_cascade.detectMultiScale(roi_gray)

            # If no noses are detected, the person is wearing a mask
            if len(noses) == 0:
                cv2.putText(frame, 'Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(frame, 'No Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the output frame
        cv2.imshow('Face Mask Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to start face mask detection using webcam
detect_face_mask()
