import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split





def load_and_preprocess_data(
    data_dir: str, 
    target_size:tuple = (224, 224) 
    ,validation_extension : tuple = ('.jpeg', '.jpg', '.bmp', '.png')
    ):
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified data directory '{data_dir}' does not exist.")
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found classes: {class_names} , with len : {len(class_names)}")
    
    corruputed_count = 0
    skipped_count = 0
    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        file_list = os.listdir(class_dir)
        
        print(f"Processing class '{label}' with {len(file_list)} files.")
        
        for file_name in file_list:
            img_path = os.path.join(class_dir,file_name)
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in validation_extension:
                try:
                    os.remove(img_path)
                    print(f"Removed file '{img_path}' due to invalid extension '{ext}'.")
                except Exception as e:
                    print(f"Error removing file '{img_path}': {e}")
                skipped_count += 1
                continue
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"[CORRUPTED] Could not read image '{img_path}'. Skipping.")
                    corruputed_count += 1
                    os.remove(img_path)
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"[ERROR] An error occurred while processing image '{img_path}': {e}. Skipping.")
                if os.path.exists(img_path):
                    os.remove(img_path)
                skipped_count += 1

    print(f"Preprocessing complete. Corrupted images: {corruputed_count}, Skipped images: {skipped_count}.")
    
    images , labels = shuffle(images, labels, random_state=42)
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    print(f"Final dataset shape: images {images_arr.shape}, labels {labels_arr.shape}")
    return images_arr, labels_arr ,class_names

def plot_class_visualization(lavels_arr):
    class_names, counts = np.unique(lavels_arr, return_counts=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=counts)
    plt.title("Class Distribution")
    plt.xlabel("Class Names")
    plt.ylabel("Counts")
    
    for i, count in enumerate(counts):
        plt.text(i, count + 20, str(count), fontsize=11)
        
        
        

def display_sample_images(images_arr, Labels_arr, class_names):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i,cls in enumerate(class_names):
        sample_index = np.where(Labels_arr == cls)[0][0]
        axes[i].imshow(images_arr[sample_index])
        axes[i].set_title(cls)
        axes[i].axis('off')
    plt.suptitle("Sample Images from Each Class")
    plt.tight_layout()
    plt.show()
    

def check_class_balance(labels_arr):
    classes, counts = np.unique(labels_arr, return_counts=True)
    total_images = len(labels_arr)
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count
    for cls, count in zip(classes, counts):
        percentage = (count / total_images) * 100
        print(f"Class '{cls}': {count} images ({percentage:.2f}%)")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("Warning: Significant class imbalance detected.")
    else:
        print("Class distribution is relatively balanced.")
    
def encoder_labels(labels_arr):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels_arr)
    return encoded_labels, le


def split_classes(images_arr, y_encoded, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(images_arr, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded)
    return X_train, X_test, y_train, y_test

def normalize_pixel_value(X_train, X_test):
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    return X_train_norm, X_test_norm

if __name__ == "__main__":
    data_directory = "brainscan_ai/data/raw" 
    load_and_preprocess_data(data_directory)