import cv2

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform object recognition
def detect_objects(frame):
    # Convert the frame to grayscale (for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

# Main function
def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform object recognition
        if ret:
            frame_with_objects = detect_objects(frame)

            # Display the resulting frame
            cv2.imshow('Object Recognition', frame_with_objects)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()