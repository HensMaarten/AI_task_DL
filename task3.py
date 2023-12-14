import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import streamlit as st
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report


# Function to load and display a few images from a given class
def display_images(class_path, num_images=3):
    images = os.listdir(class_path)[:num_images]
    plt.figure(figsize=(12, 6))
    for i, img_name in enumerate(images, start=1):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        plt.subplot(1, num_images, i)
        plt.imshow(img)
        plt.title(f"Image {i} of {class_path[8:]}")
        plt.axis('off')
    st.pyplot(plt)
    plt.close()

# Function to perform EDA on the dataset
def perform_eda(dataset_path):
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        num_images = len(os.listdir(class_path))
        class_counts[class_name] = num_images

        # Display a few images from each class
        display_images(class_path)


    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    st.pyplot(plt)
    plt.close()

def initiate_cnn(epochs=25,filters=26,dropout=0.3,reg_strength=0.05):
    NUM_CLASSES = 5

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
      layers.Conv2D(filters, (3, 3), input_shape = (64, 64, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(dropout),
      layers.Conv2D(filters, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(dropout),
      layers.Flatten(),
      layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
      layers.Dense(NUM_CLASSES, activation="softmax") 
    ])

    # Compile and train your model as usual
    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    train_network(model,epochs)


    
def train_network(model,amnt_epochs=25):
    train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                       rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_val_datagen.flow_from_directory('./resources/train/',
                                                     subset='training',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'categorical') 
    
    validation_set = train_val_datagen.flow_from_directory('./resources/train/',
                                                     subset='validation',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory('./resources/test/',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    
    history = model.fit(training_set,
                validation_data = validation_set,
                epochs = amnt_epochs
                )
    show_results(model,history,test_set)

def show_results(model,history,test_set):
    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the figure
    st.pyplot(fig)

    y_pred = model.predict(test_set)

    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(y_pred, axis=1)
    
    # Get true labels from the test set generator
    true_labels = test_set.classes
    
    # Get class names from the test set generator
    class_names = list(test_set.class_indices.keys())
    
    # Compute the confusion matrix with class names
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
    
    # Display the confusion matrix with class names
    st.title("Confusion Matrix:")
    st.text(conf_matrix)
    
    plt.close()

    # Display a heatmap of the confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    conf_matrix_filename = "confusion_matrix.png"
    plt.savefig(conf_matrix_filename)

    st.image(conf_matrix_filename)
    
    # Print classification report
    st.text("Classification Report:")
    st.text(classification_report(true_labels, predicted_labels, target_names=class_names))



st.title("Task Deep Learning")
st.subheader("By Maarten Hens")
st.text("Training a neural network to detect different kinds of food")

st.header("EDA:")
perform_eda("./resources/train/")

st.header('Select the amount of epochs!')
number_of_epochs = st.slider('Select a value:', 5, 40, 25, step=1)

st.header('Select the number of filters you want to use in the cnn!')
number_of_filters = st.slider('Select a value:', 15, 50, 26, step=1)

st.header('Select the dropout rate you want to use in the cnn!')
dropout_rate = st.slider('Select a value:', 0.1, 0.9, 0.3, step=0.1)

st.header('Select the regularization strength you want to use in the cnn!')
reg_strength = st.slider('Select a value:', 0.00, 0.50, 0.05, step=0.01)


if st.button('Run predictions'):
    initiate_cnn(number_of_epochs,number_of_filters,dropout_rate,reg_strength)
