from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import mlflow
from PIL import Image
import time
import keras
import base64
from fastapi.middleware.cors import CORSMiddleware
from keras.src.applications.efficientnet import preprocess_input

# ------------------------------------------------------
# 1. Inicialización de FastAPI
# ------------------------------------------------------

#mlflow.set_tracking_uri("http://localhost:5010")
#mlflow.set_experiment("inference_birds")
base = "/Users/lvillazonest/Downloads/CUB_200_2011/CUB_200_2011/images/"

app = FastAPI(title="Bird Species Classifier API",
              description="API para inferencia del modelo Keras entrenado en CUB-200-2011",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 2. Cargar el modelo Keras entrenado
# ------------------------------------------------------
MODEL_PATH = "/Users/lvillazonest/Downloads/best_finetuned_model.keras"
model = keras.saving.load_model(MODEL_PATH)

# Si tienes las etiquetas:
CLASS_NAMES = [f"class_{i}" for i in range(200)]  # sustituir por las reales


def retrieve_data_from_mysql(ids:list):
    from sqlalchemy import create_engine
    import pandas as pd

    usuario = "user_user"
    password = "user_birds"
    host = "127.0.0.1"
    puerto = 3306
    db = "birds"

    engine = create_engine(f"mysql+pymysql://{usuario}:{password}@{host}:{puerto}/{db}")
    ids = [str(i) for i in ids]
    df = pd.read_sql(f"""
                     SELECT *
                     FROM dim_images
                     WHERE class_id in({",".join(ids)})
                     """, con=engine)
    return df


# ------------------------------------------------------
# 3. Función de preprocesamiento
# ------------------------------------------------------
def preprocess_image(image: Image.Image, img_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image)
    return preprocess_input(np.expand_dims(img_array, axis=0))


def mlflow_metrics(file, confidence, inference_time, pred):

    with mlflow.start_run(run_name="inference"):
        mlflow.log_param("filename", file.filename)
        mlflow.log_param("content_type", file.content_type)

        with open("tmp_image.jpg", "wb") as f:
            f.write(np.array(file.read()))

        # Si quieres guardar los bytes originales
        mlflow.log_artifact(local_path="tmp_image.jpg")

        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("inference_time_ms", inference_time * 1000)

        # Puedes guardar la predicción como "tag"
        mlflow.set_tag("prediction", pred)

# ------------------------------------------------------
# 4. Endpoint para predicción
# ------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        img = Image.open(file.file)
        input_tensor = preprocess_image(img)

        preds = model.predict(input_tensor)
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        inference_time = time.time() - start_time


        df = retrieve_data_from_mysql([pred_idx])
        prediction = df[df['entry_id'] == pred_idx]['entry_name'][0]
        image = df[df['entry_id'] == pred_idx].sample(1)['image_path'].values[0]

        with open(f"{base}{image}", "rb") as f:
            imagen_bytes = f.read()

        #mlflow_metrics(file=file, confidence=confidence, inference_time=inference_time,pred=prediction)

        return JSONResponse({
            "predicted_class": prediction,
            "class_id": int(pred_idx),
            "confidence": round(confidence, 4),
            "image": base64.b64encode(imagen_bytes).decode("utf-8")
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def get_image(df, idx):
    image = df[df['class_id'] == idx].sample(1)['image_path'].values[0]
    with open(f"{base}{image}", "rb") as f:
        imagen_bytes = f.read()
    return  base64.b64encode(imagen_bytes).decode("utf-8")

@app.post("/predict-top5")
async def predict_top5(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        input_tensor = preprocess_image(img)

        preds = model.predict(input_tensor)[0]

        # Obtener top5
        top5_idx = [i for i in preds.argsort()[-5:][::-1]]
        df = retrieve_data_from_mysql(top5_idx)
        top5 = []
        for i in top5_idx:
            image = get_image(df, i)
            top5.append(
            {
                "class_id": int(i),
                "class_name": df[df['class_id'] == i]['entry_name'].values[0],
                "confidence": float(preds[i]),
                "image": image
            })
        return {"top5_predictions": top5}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
# ------------------------------------------------------
# 5. Ejecutar localmente
# ------------------------------------------------------
# uvicorn app.main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
