This is a program called Full Body Landmarks Detection
First, the necessary libraries including OpenCV and Mediapipe are imported to process and detect landmarks of the body, face and hands
A Kalman filter class is implemented which is used to reduce noise in detecting the location of the points
The camera is activated and enters a loop which receives and processes the video frames
Using mp_pose.Pose, key points of the body (landmarks) are identified
The relationships between the points of the body (bone connections) are drawn
Important angles of the body such as the angles of the knees, elbows, neck and hips are calculated
mp_hands.Hands is used to detect and display the position of the fingers
The connecting lines between the joints of the hands are drawn
The points of the fingers are displayed as red and the lines are green
mp_face_mesh.FaceMesh is used to detect 468 points on the face
A white mask with 30% transparency is applied to the face to give it a more professional look be
Calculated joints
Neck: Angle between two shoulders and nose.
Right and left elbows: Angle between shoulder, elbow and wrist.
Right and left knees: Angle between hip, knee and ankle.
Hip: Angle between two hip joints and one knee.
Kalman filter to improve accuracy
Due to noise in point detection, the Kalman filter is used to better predict the location of landmarks.
This filter smoothes the current value of each point with a predicted value to reduce sudden jumps.
Display and save output
Two windows are displayed:
Original Frame (raw camera image)
Processed Landmarks (image on which landmarks and angles are displayed)
In each frame, the angles and positions of the joints are displayed on the image.
If the ESC key is pressed, the program stops and the windows are closed.
Applications of the program
✅ Biomechanical analysis: To examine how joints move in athletes and patients.
✅ Diagnosis of posture abnormalities: such as scoliosis, kyphosis and skeletal problems.
✅ Analysis of neurological diseases: such as Parkinson's and Bell's palsy by analyzing movement patterns.
✅ Evaluation of hand movements: for the rehabilitation of patients after brain injuries or strokes.
✅ Use in robotics and artificial intelligence: to control robots based on body movements.
Conclusion
This program is an advanced model for recognizing body, hand and face landmarks that can be used in medicine, sports, rehabilitation and research. The Kalman filter is used to increase accuracy and optimal methods are used to calculate angles. I started this program from the land _mark file and the land _mark _10 file is the last one.
