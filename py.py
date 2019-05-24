import cv2
import time
import datetime
import math
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'pbpb.pbtxt')
video_capture = cv2.VideoCapture("s.mp4")
i=0
zeros=set()
ones=set()
classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

while(video_capture.isOpened()):
    # Input image
    # Check success
    a=datetime.datetime.now()
    if not video_capture.isOpened():
       raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, img = video_capture.read()
    # Close device
    rows, cols, channels = img.shape

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0,0]:

        score = float(detection[2])
        if score > 0.4:
            #print(detection)
            print(classes_90[int(detection[1])])
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.putText(img, classes_90[int(detection[1])], (int(left), int(top)-15),0, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            #draw a red rectangle around detected objects
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    # Show the image with a rectagle surrounding the detected objects 
    #cv2.imshow('Image', img)
    b=datetime.datetime.now()
    dur=b-a
    print(round(1/dur.total_seconds()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

video_capture.release()
cv2.destroyAllWindows()

