import tensorflow as tf
import numpy as np

IMG_SIZE = (260, 260)

CLASS_NAMES = [
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
    "Tuberculosis"
]

# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model():
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=(260, 260, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(260, 260, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


# -----------------------------
# LOAD WEIGHTS
# -----------------------------
model = build_model()
model.load_weights("models/efficientnetb2_6class_260px_final_v2.keras/model.weights.h5")

print("✅ Chest model loaded successfully")


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_chest_xray(pil_image):
    if pil_image is None:
        return None

    # Convert to grayscale first (because training used grayscale)
    pil_image = pil_image.convert("L")

    # Resize
    pil_image = pil_image.resize(IMG_SIZE)

    # Convert to array
    img = tf.keras.preprocessing.image.img_to_array(pil_image)

    # Convert grayscale → RGB
    img = np.repeat(img, 3, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # EfficientNet preprocessing
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    probs = model.predict(img)[0]
    idx = np.argmax(probs)

    return {
        "disease": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(6)
        }
    }