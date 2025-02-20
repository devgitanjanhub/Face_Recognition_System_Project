import cv2

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.xml")  # Path to the trained model file

# Load labels (names of people) for recognition
labels = {"person_name": 1}  # You need to specify your labels here

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Recognize the face
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 (0 is perfect match)
        if confidence < 100:
            # Get the label of the recognized person
            name = labels[id_]
            confidence_text = f"Confidence: {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"Confidence: {round(100 - confidence)}%"

        # Put text label on the recognized face
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, confidence_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
