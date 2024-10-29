import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the camera frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale (black and white)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale frame to BGR format so we can stack them horizontally
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Create a spacer (a black frame to separate the two frames horizontally)
    spacer_horizontal = np.zeros((frame.shape[0], 10, 3), dtype=np.uint8)  # 10 pixels wide black space
    
    # Stack the original frame and grayscale frame horizontally with a spacer
    combined_frames_horizontal = np.hstack((frame, spacer_horizontal, gray_frame_bgr))
    
    # Create a vertical spacer (a black rectangle) for extra space below
    spacer_vertical = np.zeros((50, combined_frames_horizontal.shape[1], 3), dtype=np.uint8)  # 50 pixels tall black space
    
    # Stack the combined frames and vertical spacer vertically
    final_output = np.vstack((combined_frames_horizontal, spacer_vertical))
    
    # Display the combined frame with vertical space
    cv2.imshow('Combined Frame with Vertical Space', final_output)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
