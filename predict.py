# ==========================================
# FACE MAKEUP DETECTION - PREDICTION SCRIPT
# ==========================================

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("makeup_model.h5")

img = image.load_img("test.jpg", target_size=(224,224))
img_array = image.img_to_array(img)/255
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: NO MAKEUP")
else:
    print("Prediction: MAKEUP")
