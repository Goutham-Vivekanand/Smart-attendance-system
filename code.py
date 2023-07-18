from PIL import Image
import numpy as np
import cv2
import pickle
import csv
import tensorflow as tf

# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 224
image_height = 224

# load the trained model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# the labels for the trained model
with open("face-labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    print(labels)

# default webcam
stream = cv2.VideoCapture(0)
register = []
filename = "names.csv"

while True:
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each face found
    for (x, y, w, h) in faces:
        roi_rgb = rgb[y:y+h, x:x+w]

        # Draw a rectangle around the face
        color = (255, 0, 0)  # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        # resize the image
        size = (image_width, image_height)
        resized_image = cv2.resize(roi_rgb, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1, image_width, image_height, 3)
        img = img.astype('float32')
        img /= 255

        # Perform inference with TensorFlow Lite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predicted_prob = interpreter.get_tensor(output_details[0]['index'])

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[predicted_prob[0].argmax()]
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x, y-8),
                    font, 1, color, stroke, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Image", frame)

        if x < 20:
            print(f"{name} - exited")
            if x > 400:
                with open(filename, 'a', newline='') as csvfile:
                    if name not in register:
                        register.append(name)
                        writer_object = csv.writer(csvfile)
                        writer_object.writerow([name])
                        csvfile.close()
            else:
                print(f"{name} was a proxy")

    if cv2.waitKey(1) == ord('q'):  # Press q to break out of the loop
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
