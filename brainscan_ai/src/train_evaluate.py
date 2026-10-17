
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def show_prediction_image(indices,title,y_true, y_pred, normalized_X_test, class_names):

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices[:5]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(normalized_X_test[idx])
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()
    


def predict_image(path):
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
    img = cv2.resize(img, (224, 224))               
    img = img.astype('float32') / 255.0             
    img = np.expand_dims(img, axis=0)      

    model = tf.keras.models.load_model("brain_tumor_detector.h5")
    predictions = model.predict(img)

    predicted_index = np.argmax(predictions[0])
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_label = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100
    
    print(f"Diagnostic : {predicted_label} ({confidence:.2f}%)")
    return predicted_label, confidence