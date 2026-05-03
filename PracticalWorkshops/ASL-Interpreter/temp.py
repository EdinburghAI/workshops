# # Workshop 2: Using Computer Vision for Good! 🚀
# 
# - **When:** Wednesday, February 26th, 17:00.
# - **Where:** Appleton Tower, 5.04. If you don't have access, reach out and we'll let you in!
# - **Contacts:** Reach out on Instagram _@edinburgh.ai_
# - **Credits:** This notebook is created by EdinburghAI for use in its workshops. If you plan to use it, please credit us!
# 
# ## Today
# - Today we're building a Sign Language Interpreter. 
# - We'll take hundreds of photos of differenxt hand signs and extract the relevant information.
# - We'll try training different models.
# - We'll then upload our own hand sign and see if our model accurately predicts it!
# 
# Lfg 🔥

# # Setup:
# 
# ## IMPORTANT! Turn On Internet
# 1. On the right-hand side of this notebook, there's a section called **"Session Options"**.
# 2. Scroll down to the _"Internet"_ toggle. Turn it on. You may need to verify your phone number.
# 3. Additionally, to help this run faster, you can also enable some GPU access.
# 
# 
# ## Using Jupyter:
# This is a Jupyter notebook. It contains cells. There are 2 kinds of cells - markdown and Python. Markdown cells are like this one, and are just there to give you information. Python cells run code. You can run a cell with `Ctrl + Enter` or run a cell and move the next one with `Shift + Enter`. Try running the cell below.

print('Ctrl + Enter runs this cell!')
output = 'The last line of a cell is printed by default'
output

# Installation
#!pip install mediapipe
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# # Exploring Our Data
# 
# Like all good AI projects, we need to explore our data first. Check it out!

### First, let's take a look at the data. 

# We'll be using OpenCV for this project.
# OpenCV is a popular library for computer vision.
import cv2

# Let's load in an image and display it.
img = cv2.imread('/kaggle/input/synthetic-asl-alphabet/Train_Alphabet/A/0042513a-63c0-499f-a7f7-e6ee1266cb98.rgb_0000.png')
plt.imshow(img)


# Hmmmm. That doesn't look quite right. Turns out OpenCV defaults to BlueGreenRed, but our eyes do not lol. 
# 
# 🤔 _Why do you think that is?_ 
# 
# Regardless, let's turn it to RGB. 

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

# That's better! Ok sweet. We're gonna do a few things from here. 
# - We're gonna take all of the crucial parts of the hand (wrist, phalanges lol, etc.) and then feed _that_ into a model. 
# - We're gonna (inefficiently) feed the entire image into a neural network and pray. 
# 

# ## Extracting Key Features of the Hand
# 
# #### Ok... How?
# We're gonna use a library called _MediaPipe_. Some really smart people made a suite of models - we're going to use the _Hands_ model. This will try and identify if there are any hands in the photo. If there are, we'll place coordinates on top of the photo. 
# 
# Let's try it out!
# 
# ### Quickly, what will it look like?
# Let's use the MediaPipe Drawing Utils to try and draw 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

hand_landmarks = hands.process(img_rgb)

for landmark in hand_landmarks.multi_hand_landmarks:
    
    mp_drawing.draw_landmarks(
            img_rgb,  # image to draw
            landmark,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    plt.imshow(img_rgb)


# ### Great. Now get those landmarks for every image.
# 
# Sweet. So we found out that `mp.solutions.hands` finds an image and turns it the landmarks of a hand into a bunch of coordinates. 
# 
# _Also as a side note, we're going to normalise them so all of the coordinates are relative to 0 (as opposed to where they actually were in the photo)._

import os
import os
import cv2
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar


DATA_DIR = "/kaggle/input/synthetic-asl-alphabet/Train_Alphabet"

## MAKE DATA FROM NORMAL IMAGES
def turnImagesToLandmarks(DATA_DIR):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    
    data = []
    labels = []
    for dir_ in (os.listdir(DATA_DIR)):
        for img_path in tqdm(os.listdir(os.path.join(DATA_DIR, dir_))[:30]):
            data_aux = []
    
            x_ = []
            y_ = []
    
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Only take the first hand
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                data.append(data_aux)
                labels.append(dir_)

        print(f"Finished letter {dir_}")
    return labels, data
    


labels, data = turnImagesToLandmarks(DATA_DIR)

# Ummm. Did that work? Let's take a look at the what we made...

# Let's print out only the first data point and first label.
data[0], labels[0]

# ### Sweet! That seems right...
# Now let's train a model. Let's use RandomForest. Don't forget to split the data into a corresponding train-test split and then get the accuracy. 
# 
# If all of that seems like non-sense, take a look at our first workshop we went [through in Sem1](https://github.com/EdinburghAI/workshops/blob/main/Sem1Workshop1/IntroToML/IntroToML-Solved.ipynb) - Scroll down to _"Decision Trees"_. 

import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle  # For model saving & loading

# ✅ Load your trained data (assuming `data` & `labels` are already prepared)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# ✅ Train the Random Forest Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# ✅ Evaluate Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# ✅ Save Model for Later Use
with open('asl_hand_model.pkl', 'wb') as f:
    pickle.dump(model, f)




# ### That was fast! 
# The model trained reallyyyy quickly. In ML, you often don't have to pick the fanciest model. 
# 
# We're basically taking coordinates and trying to find a shape between them - therefore our model doesn't need to be the most complicated. This makes everything wayyy faster. 

# ### Inference Time
# Now let's test it out! Take a photo on your phone of you doing a Sign and see if the model will recognise it!
# 
# 
# To do this:
# - Take a photo on your phone. Ensure there's only 1 hand visible in the frame.
# - Pass the photo to your laptop.
# - On the right hand side, scroll to the _"Upload"_ button > _"New Dataset"_. 
# - Give it a name
# - Press create!

# -------------------------
# 🚀 Inference Function
# -------------------------
def infer_single_image(img_path, model_path='asl_hand_model.pkl'):
    """
    Takes an image path, extracts hand landmarks, and predicts the ASL letter using the trained model.
    """
    # ✅ Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ✅ Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # ✅ Load and preprocess image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only take the first detected hand
        x_ = np.array([lm.x for lm in hand_landmarks.landmark])
        y_ = np.array([lm.y for lm in hand_landmarks.landmark])

        # ✅ Normalize features (same as training)
        x_min, y_min = x_.min(), y_.min()
        data_aux = np.column_stack((x_ - x_min, y_ - y_min)).flatten().tolist()

        # ✅ Predict using trained model
        prediction = model.predict([data_aux])[0]

        return prediction  # Return predicted label
    else:
        return "No hand detected!"


# -------------------------
# 🚀 Example Usage
# -------------------------
img_path = "/kaggle/input/randomphotoofa/IMG_4192.JPG"  # Replace with your actual test image
predicted_label = infer_single_image(img_path)
print(f"Predicted ASL Letter: {predicted_label}")


