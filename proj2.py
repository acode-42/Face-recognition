import face_recognition
import cv2
import numpy as np

# Load known face images for 5 people
person1_image = face_recognition.load_image_file("me1.jpg")
person2_image = face_recognition.load_image_file("Nitin.jpg")
person3_image = face_recognition.load_image_file("Sarthak.jpg")
person4_image = face_recognition.load_image_file("Raghav.jpg")
person5_image = face_recognition.load_image_file("Shubham.jpg")

# Get face encodings for each person
person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_encoding = face_recognition.face_encodings(person2_image)[0]
person3_encoding = face_recognition.face_encodings(person3_image)[0]
person4_encoding = face_recognition.face_encodings(person4_image)[0]
person5_encoding = face_recognition.face_encodings(person5_image)[0]

# Create arrays of known face encodings and names
known_face_encodings = [
    person1_encoding, 
    person2_encoding, 
    person3_encoding, 
    person4_encoding, 
    person5_encoding
]
known_face_names = [
    "Tanay", 
    "Nitin", 
    "Sarthak", 
    "Raghav", 
    "Shubham"
]

# Initialize video capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the application.")

while True:
    # Capture frame
    ret, frame = video_capture.read()
    if not ret:
        print("Warning: Failed to capture a frame.")
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale face locations back to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Check for matches
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()