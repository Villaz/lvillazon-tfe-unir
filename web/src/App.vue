<template>
  <div class="page">
    <h1> Inferencia de Aves</h1>
    <!-- Subir imagen -->
    <input type="file" accept="image/*" @change="onFileChange" />

    <!-- Mostrar imagen subida -->
    <div v-if="uploadedImage" class="uploaded-container">
      <h3>Imagen enviada:</h3>
      <img :src="uploadedImage" class="uploaded-img" />
    </div>

    <!-- Botón -->
    <button
        :disabled="!file"
        @click="sendImage"
        class="send-btn"
    >
      Enviar imagen
    </button>

    <!-- Resultados -->
    <div v-if="predictions.length > 0" class="results-section">
      <h3>Top 5 inferencias:</h3>

      <div class="grid-container">
        <div v-for="item in predictions" :key="item.class_id" class="card">
          <img :src="item.image_data_uri" class="fixed-img" />

          <div class="info">
            <h4>{{ item.class_name }}</h4>
            <p>{{ (item.confidence * 100).toFixed(2) }}%</p>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>


<script setup>
import { ref } from "vue";

const file = ref(null);
const uploadedImage = ref(null);
const predictions = ref([]);

// Se llama cuando el usuario sube una imagen
function onFileChange(event) {
  file.value = event.target.files[0];

  // Mostrar preview
  uploadedImage.value = URL.createObjectURL(file.value);
}

// Enviar la imagen al endpoint
async function sendImage() {
  const formData = new FormData();
  formData.append("file", file.value);

  const res = await fetch("http://localhost:8000/predict-top5", {
    method: "POST",
    body: formData,
  });

  predictions.value = await res.json();
  predictions.value = predictions.value.top5_predictions
  console.log(predictions);
  console.log("Longitud de Predictions después de la asignación:", predictions.value.length);

  predictions.value = predictions.value.map(item => {
    // Asumo que la clave que contiene los datos binarios se llama 'image_data_base64'
    // o simplemente 'image' si tu backend ya la codifica en Base64.
    // SI TU BACKEND NO CODIFICA A BASE64, ESTO FALLARÁ.

    // Suponemos que el backend YA ha codificado los bytes binarios a Base64.
    // El formato completo es: data:tipo/subtipo;base64,DATOS
    const dataUri = `data:image/jpeg;base64,${item.image}`;

    return {
      ...item, // Mantiene class_id, class_name, confidence, etc.
      image_data_uri: dataUri // Nueva clave con la URL Base64
    };
  });
  console.log("Hecho")
}
</script>


<style scoped>
.page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Imagen enviada */
.uploaded-container {
  margin-top: 10px;
}

.uploaded-img {
  width: 250px;
  height: auto;
  border-radius: 10px;
}

/* Botón */
.send-btn {
  width: 200px;
  padding: 10px;
  cursor: pointer;
}

/* Resultados */
.grid-container {
  margin-top: 20px;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 15px;
}

.fixed-img {
  width: 200px;
  height: 200px;
  object-fit: cover;
  border-radius: 10px;
  border: 2px solid #aaa;
}

.card {
  text-align: center;
}

.info {
  margin-top: 10px;
}
</style>
