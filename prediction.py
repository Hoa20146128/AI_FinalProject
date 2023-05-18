from email.mime import message
import cv2
import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from PIL import ImageTk ,Image
from matplotlib.pyplot import show
import numpy as np
from tkinter import messagebox
from tensorflow import keras
from keras.utils import load_img
from keras.utils.image_utils import img_to_array

#Load Model
model = keras.models.load_model('fashion.h5') 

# Create the GUI window
from tkinter import filedialog
window = Tk()
window.title("Image recognition for fashion industry")
window.geometry('800x500')

lbl = tk.Label(window, text="IMAGE RECOGNITION FOR FASHION INDUSTRY ", fg = "white", bg = "black", font = ("Arial",20), width= 50, pady = 20)
lbl.pack(side = TOP, fill = None, expand= False)

# Create a label to display the prediction result
result_label = tk.Label(window, text="Prediction: ",fg="blue", font=("Arial", 18), pady = 20)
result_label.pack(side= BOTTOM, fill= Y, expand= False)

# Function to perform lung disease prediction
def predict_disease(image_path):
    img = Image.open(image_path).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the loaded model
    prediction = model.predict(img_array)
    # Assuming the model outputs one-hot encoded labels, convert prediction to class labels
    class_labels = ['Jacket', 'Pant', 'Shoe','T-shirt']
    predicted_label = class_labels[np.argmax(prediction)]
    result_label.config(text="Prediction: " + predicted_label)

# Function to handle button click event
def open_image():
    image_path = filedialog.askopenfilename(initialdir="test_images", title="Select Image",
                                            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))                                      
    predict_disease(image_path)
    
    # Display the selected image in the GUI
    img = Image.open(image_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(window, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady = 20)

# Create a button to open an image for prediction
open_button = tk.Button(window, text="Open Image", command=open_image, font=("Arial", 18), bg = "green")
open_button.pack()

window.mainloop()