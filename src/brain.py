import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary Tumor",
    "No Tumor"
]

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None, 
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


# 🔥 Build and load weights
model = build_model()
model.load_weights("models/brain.keras/model.weights.h5")


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return img_array


def predict_brain_tumor(pil_image):
    if pil_image is None:
        return None

    pil_image = pil_image.resize(IMG_SIZE)

    img = tf.keras.preprocessing.image.img_to_array(pil_image)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    probs = model.predict(img)[0]
    idx = np.argmax(probs)

    return {
        "disease": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(4)
        }
    }