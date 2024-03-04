#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 01:58:50 2024

@author: beyzanurokudan
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam

# Veri seti yolu ve sınıflar
train_dir = 'base_dir/image_train'
validation_dir = 'base_dir/image_val'
test_dir = 'base_dir/image_test'
num_classes = 7  # Toplam sınıf sayısı

# ImageDataGenerator ile veri artırma ve ön işleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Veri setlerini yükleme
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 75),
    batch_size=20,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 75),
    batch_size=20,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 75),
    batch_size=20,
    class_mode='categorical')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

input_shape = (75, 100, 3)  # Örnek olarak, girdi şeklinizi ayarlayın
num_classes = 7  # Sınıf sayınız

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')

epochs = 50  # Artırılan epoch sayısı
batch_size = 32  # Batch boyutu

# Model eğitimi
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Test seti üzerinde model değerlendirme
test_loss, test_acc = model.evaluate(test_generator, steps=50)  # Test setindeki toplam resim sayısına bağlı olarak ayarlayın
print('Test accuracy:', test_acc)


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

# Önceden eğitilmiş VGG16 modelini yükle
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(100, 75, 3))

# Önceden eğitilmiş katmanların ağırlıklarını dondur
for layer in base_model.layers:
    layer.trainable = False

# Yeni modeli oluştur
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Modeli derle
model = Model(inputs=base_model.input, outputs=predictions)
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Model özeti
model.summary()

# Model eğitimi
epochs = 30  # Artırılan epoch sayısı
batch_size = 32  # Güncellenmiş batch boyutu

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=1
)

# Modeli kaydet
model.save('vgg16_finetuned_model.h5')


# GUI ve diğer fonksiyonlarınız burada devam edebilir
# [...]
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np

# Modeli yükle
model = load_model('vgg16_finetuned_model.h5')

# Sınıf etiketleri
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

img_path = ''  # Resmin yolu için global değişken

from PIL import Image, ImageTk
import numpy as np

def load_image(img_path, target_size=(100, 75)):
    img = Image.open(img_path)
    # Resmi doğru boyutlara yeniden boyutlandır
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Güncellenmiş yöntem
    img = np.array(img)
    # Eğer girdi boyutu modelin beklediği ile uyuşmuyorsa, boyutları değiştir
    if img.shape != (100, 75, 3):
        img = np.transpose(img, (1, 0, 2))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img



def classify_image():
    global img_path
    if img_path:
        img = load_image(img_path)
        prediction = model.predict(img)
        class_id = np.argmax(prediction)
        class_label = class_labels[class_id]
        messagebox.showinfo("Sonuç", f"Bu resim muhtemelen aşağıdaki sınıfa ait: {class_label}")
    else:
        messagebox.showerror("Hata", "Lütfen bir resim seçin")

def open_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename(title="Resim Seçin",
                                          filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    if img_path:
        img = Image.open(img_path)
        img = img.resize((250, 250), Image.LANCZOS)
        img_display = ImageTk.PhotoImage(img)  # img_display global değişkeninde sakla
        panel.configure(image=img_display)
        panel.image = img_display  # Bu satırı eklemek panel üzerindeki referansı korur

# GUI
window = tk.Tk()
window.geometry("500x500")
window.title("Deri Kanseri Sınıflandırma")

# Resim paneli
panel = tk.Label(window)
panel.pack(pady=20)

# Butonlar
btn_open = tk.Button(window, text="Resim Aç", command=open_image)
btn_open.pack(pady=5)

btn_classify = tk.Button(window, text="Sınıflandır", command=classify_image)
btn_classify.pack(pady=5)

window.mainloop()
