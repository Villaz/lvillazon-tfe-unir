<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center p-6 gap-6">
    <h1 class="text-3xl font-bold">Bird Classifier - Top 5 (Vue)</h1>

    <div class="w-full max-w-xl bg-white p-4 rounded-2xl shadow">
      <div class="flex flex-col gap-4">
        <input type="file" accept="image/*" @change="onFileChange" />

        <img
            v-if="preview"
            :src="preview"
            alt="preview"
            class="w-full rounded-2xl shadow"
        />

        <button
            @click="sendImage"
            :disabled="loading || !image"
            class="px-4 py-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-400"
        >
          {{ loading ? "Procesando..." : "Enviar imagen" }}
        </button>
      </div>
    </div>

    <div v-if="results.length > 0" class="w-full max-w-xl bg-white p-4 rounded-2xl shadow">
      <h2 class="text-xl font-semibold mb-4">Top 5 Predicciones</h2>
      <ul class="space-y-2">
        <li
            v-for="(item, idx) in results"
            :key="idx"
            class="p-3 bg-gray-50 rounded-xl shadow flex justify-between"
        >
          <span class="font-medium">{{ item.class_name }}</span>
          <span>&nbsp;&nbsp;</span>
          <span> {{ (item.confidence * 100).toFixed(2) }}%</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: "Top5PredictorVue",
  data() {
    return {
      image: null,
      preview: null,
      results: [],
      loading: false,
      API_URL: "http://localhost:8000/predict-top5",
    };
  },
  methods: {
    onFileChange(e) {
      const file = e.target.files[0];
      if (file) {
        this.image = file;
        this.preview = URL.createObjectURL(file);
      }
    },
    async sendImage() {
      if (!this.image) return;

      this.loading = true;
      const formData = new FormData();
      formData.append("file", this.image);

      try {
        const res = await fetch(this.API_URL, {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        console.info(data);
        this.results = data.top5_predictions || [];
      } catch (err) {
        console.error(err);
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style>
body {
  font-family: Arial, sans-serif;
}
</style>
