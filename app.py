import cv2
import numpy as np

# Define a known real-world width of the reference object in centimeters (e.g., a coin of 2.5 cm diameter)
REFERENCE_OBJECT_WIDTH_CM = 2.5  # Adjust this to your reference object size

# Function to calculate the pixel-to-real-world ratio using a reference object
def calculate_pixel_to_metric_ratio(reference_width_px, real_world_width_cm):
    return real_world_width_cm / reference_width_px

# Function to process the frame and detect objects
def process_frame(frame, pixel_to_metric_ratio=None):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image to separate objects from the background
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the objects in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and calculate the area and bounding box
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Only proceed if the area is above a threshold to avoid noise
        if area > 500:
            # Get bounding box around the object
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If the pixel-to-metric ratio is available, convert width and height
            if pixel_to_metric_ratio:
                object_width_cm = w * pixel_to_metric_ratio
                object_height_cm = h * pixel_to_metric_ratio
                cv2.putText(frame, f"W: {object_width_cm:.2f} cm H: {object_height_cm:.2f} cm",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Display the dimensions in pixels if no ratio is available
                cv2.putText(frame, f"W: {w} px H: {h} px", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# Function to detect the reference object and compute pixel-to-metric ratio
def detect_reference_object(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image to isolate the reference object
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours to find the reference object (assumed to be the largest contour)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Assume reference object has the largest area
        if area > 500:
            # Get bounding box for the reference object
            x, y, w, h = cv2.boundingRect(cnt)
            # We use the width of the bounding box to calculate the ratio
            pixel_to_metric_ratio = calculate_pixel_to_metric_ratio(w, REFERENCE_OBJECT_WIDTH_CM)
            # Mark the reference object on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Reference Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return frame, pixel_to_metric_ratio

    return frame, None

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Main loop to capture video and process each frame
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture video")
        break

    # Detect reference object and get the pixel-to-metric ratio
    frame, pixel_to_metric_ratio = detect_reference_object(frame)

    # Process the frame to detect and measure objects using the ratio
    frame = process_frame(frame, pixel_to_metric_ratio)

    # Display the resulting frame
    cv2.imshow('Real-Time Object Measurement', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
