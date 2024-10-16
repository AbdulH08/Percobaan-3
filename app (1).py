import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Define file ID and path
file_id = '17tPbN2_7ZyBriedo7gHEB2KEYPE90rRs'  # Gantilah dengan ID file Anda
model_path = 'modelVGG16ep24.h5'

# Fungsi untuk memuat model (gantilah 'model_path' dengan path model Anda)
@st.cache(allow_output_mutation=True)
def load_model():
    # Check if the model already exists, if not, download it
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)  # Ganti dengan path model Anda
    return model

# Memuat model
model = load_model()

# Fungsi untuk memprediksi gambar menggunakan model CNN
def predict(image, model):
    # Mengubah ukuran gambar sesuai dengan input model
    image = image.resize((224, 224))  # Sesuaikan dengan ukuran input model Anda
    # Mengubah gambar menjadi array numpy dan menormalkan
    image = np.array(image) / 255.0
    # Menambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    # Membuat prediksi
    predictions = model.predict(image)
    return predictions

# Judul aplikasi
st.title("Alat Deteksi Penyakit Mata dengan CNN")

# Mengunggah gambar
uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Membuat prediksi jika model sudah dimuat
    if model is not None:
        predictions = predict(image, model)

        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:", predictions)

        # Mendapatkan indeks kelas yang diprediksi
        predicted_class_index = np.argmax(predictions[0])

        # Menampilkan hasil prediksi dengan label
        class_labels = ['armd', 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        st.write('Kelas Prediksi:', class_labels[predicted_class_index])
    else:
        st.write("Mohon tunggu, model sedang dimuat...")

# Menjalankan aplikasi Streamlit
if __name__ == '__main__':
    st.write("Silakan unggah gambar untuk memulai deteksi.")
