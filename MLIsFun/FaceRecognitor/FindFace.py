import sys
import dlib
import cv2
# import openface # openface can only be installed in linux
from skimage import io


# take the image file name from the stored directory
file_path = sys.argv[1]

# load the pre-trained model
predictor_model = sys.argv[2]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_poss_detector = dlib.shape_predictor(predictor_model)
# face_aligner = openface.AlignDlib(predictor_model)

win = dlib.image_window()

# Load the image into an array
image = io.imread(file_path)

# Run the HOG face detector on the image data
# The result will be the bounding boxes of the faces in our image
detected_face = face_detector(image, 1)

print('I found {} face(s) in the file {}'.format(len(detected_face), file_path))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_face):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right, bottom edges
    print('- Face #{} found at Left: {} - Right: {} - Top: {} - Bottom: {}'
        .format(i, face_rect.left(), face_rect.right(), face_rect.top(), face_rect.bottom()))
    # Draw a box around each face we found
    win.add_overlay(face_rect)

    # Get the face's pose
    pose_landmarks = face_poss_detector(image, face_rect)

    # Draw faces's landmarks
    win.add_overlay(pose_landmarks)

    # Use openface to calculate and perform the face alignment
    # alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    # Save the aligned image to a file
    # cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()