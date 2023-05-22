import base64
from io import BytesIO
import PIL
from PIL import Image

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

model_checkpoint = "facebook/convnext-base-224"
model_checkpoint = "eason0203/swin-tiny-patch4-window7-224-finetuned-arty"
bg_model_checkpoint = "eason0203/swin-tiny-patch4-window7-224-arty-bg-classifier"


pipe = pipeline("image-classification", model_checkpoint, device=-1)
bg_pipe = pipeline("image-classification", bg_model_checkpoint, device=-1)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64Image(BaseModel):
    data_url: str


class SetModelDto(BaseModel):
    model_name: str


class Prediction(BaseModel):
    score: float
    label: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/test")
async def post_test():
    return JSONResponse(content={"message": "Hello World"})


@app.post("/set-model")
async def post_test(set_model_dto: SetModelDto):
    global pipe, bg_pipe
    try:
        pipe = pipeline("image-classification",
                        set_model_dto.model_name, device=-1)
    except Exception as e:
        return JSONResponse(content={"message": f"Error:\n{e}"})
    return JSONResponse(content={"message": f"set model to {set_model_dto.model_name} successfully!"})


@app.post("/predict")
async def predict(base64_image: Base64Image):
    global pipe, bg_pipe

    # Extract the base64-encoded image data
    image_data = base64.b64decode(base64_image.data_url.split(",")[1])

    # Load the image data into a PIL Image object
    try:
        image = Image.open(BytesIO(image_data))
    except PIL.UnidentifiedImageError:
        return JSONResponse(content=[])

    # Run the image through the pre-trained model
    predictions: List[Prediction] = pipe(image)
    bg_predictions: List[Prediction] = bg_pipe(image)

    is_bg = [v for v in bg_predictions if v['label'] == "bg"]

    # append at the start
    predictions = is_bg + predictions

    return JSONResponse(content=predictions)
