
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
import os
import pandas as pd
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

image_folder = '../fgvc-aircraft-2013b/data/images'
train_file = '../fgvc-aircraft-2013b/data/images_manufacturer_trainval.txt'
test_file = '../fgvc-aircraft-2013b/data/images_manufacturer_test.txt'
img_height, img_width = 64, 64  #Desired image size
df_train = pd.read_csv('../train.csv')
df_test = pd.read_csv('../test.csv')



TIME_LIMIT = 32400 - 60*10
start_time = time.time()
def elapsed_time(start_time):
    return time.time() - start_time

label_encoder = LabelEncoder()

label_map_train = dict(zip(df_train['image_id'], df_train['class']))
label_map_test = dict(zip(df_test['image_id'], df_test['class']))
img_height, img_width = (64,64)

def load_image(filename, folder, label_map, size):
    img = Image.open(os.path.join(folder, filename))
    if img is not None:
        img = img.resize(size)
        image_array = np.array(img)
        label = label_map[filename]
        return (image_array, label)
    else:
        return None

def load_color_images_from_folder_parallel(folder, label_map, size=(img_height, img_width)):
    images = []
    labels = []

    # 使用 ThreadPoolExecutor 加速图像加载
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, filename, folder, label_map, size)
                   for filename in os.listdir(folder) if filename in label_map]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                images.append(result[0])
                labels.append(result[1])

    return np.array(images), np.array(labels)


def load_all_images():
    x_train, y_train = load_color_images_from_folder_parallel(image_folder, label_map_train)
    x_test, y_test = load_color_images_from_folder_parallel(image_folder, label_map_test)

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    all_images = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    return all_images, all_labels

all_images, all_labels = load_all_images()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_images(root_dir, n_classes, images_per_class=10):
    """
    Visualize images in a grid with n_classes columns and images_per_class rows per column.

    Parameters:
        root_dir (str): Directory containing class folders with images.
        n_classes (int): Number of classes (columns in the grid).
        images_per_class (int): Number of images per class (rows per column).
    """
    fig, axes = plt.subplots(images_per_class, n_classes, figsize=(n_classes * 2, images_per_class * 2))
    fig.suptitle("Generated Images", fontsize=20)
    axes = axes.flatten()  # Flatten the axes for easy iteration
    
    classes_list = os.listdir(root_dir)
    for class_idx in range(n_classes):
        class_dir = os.path.join(root_dir, classes_list[class_idx])
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        # Load images for the current class
        images = sorted(os.listdir(class_dir))[:images_per_class]
        for row_idx, image_name in enumerate(images):
            img_path = os.path.join(class_dir, image_name)
            img = Image.open(img_path)
            ax_idx = row_idx * n_classes + class_idx  # Calculate subplot index
            axes[ax_idx].imshow(img)
            axes[ax_idx].axis("off")
            if row_idx == 0:  # Add class label to the top of the column
                axes[ax_idx].set_title(classes_list[class_idx], fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for title space
    plt.show()
