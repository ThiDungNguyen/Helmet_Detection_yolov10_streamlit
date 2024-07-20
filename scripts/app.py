
import zipfile
import gdown
import uuid
import os
import numpy as np
import base64
import streamlit as st
from ultralytics import YOLOv10

origin_weight_path = 'models/yolov10/weights/helmet_safety_best.pt'
export_weight_path = 'models/yolov10/weights/helmet_safety_best.onnx'
yaml_path = 'models/yolov10/helmet_safety.yaml'


@st.cache_data(max_entries=1000)
def export_model(origin_model_path):
    model = YOLOv10(origin_model_path)

    model.export(format='onnx',
                 opset=13,
                 simplify=True)


def download_model():
    id = '1qMkeShHvvp5zix5OGWcxcav4YPOCqu_c'ls
    unzip_dest = 'models/yolov10/weights'
    os.makedirs(unzip_dest, exist_ok=True)

    gdown.download(id=id,
                   output=origin_weight_path,
                   quiet=True,
                   fuzzy=True)

    export_model(origin_model_path=origin_weight_path)


def generate_name():
    uuid_str = str(uuid.uuid4())
    return uuid_str


def save_upload_file(upload_file, save_folder='images'):
    os.makedirs(save_folder, exist_ok=True)
    if upload_file:
        new_filename = generate_name()
        save_path = os.path.join(save_folder, new_filename)
        with open(save_path, 'wb+') as f:
            data = upload_file.read()
            f.write(data)
        return save_path
    else:
        raise ('Image not found.')


def delete_file(file_path):
    os.remove(file_path)


def process_and_display_image(image_path):
    result_img = inference(image_path,
                           weight_path=export_weight_path,
                           yaml_path=yaml_path)
    st.markdown('**Detection result**')
    st.image(result_img)


def inference(img_path: str,
              weight_path: str = 'weights/helmet_safety_best.onnx',
              yaml_path: str = 'helmet_safety.yaml') -> np.ndarray:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    detector = YOLOv10(model_path=weight_path,
                       class_mapping_path=yaml_path,
                       original_size=(w, h))
    detections = detector.detect(img)
    detector.draw_detections(img, detections=detections)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def main():

    st.title('AIO2024 - Module01 - Image Project')
    st.title(':sparkles: :blue[YOLOv10] Helmet Safety Detection Demo')

    uploaded_img = st.file_uploader(
        '__Input your image__', type=['jpg', 'jpeg', 'png'])
    example_button = st.button('Run example')

    st.divider()

    if example_button:
        process_and_display_image('static/example_img.jpg')

    if uploaded_img:
        uploaded_img_path = save_upload_file(uploaded_img)
        try:
            process_and_display_image(uploaded_img_path)
        finally:
            delete_file(uploaded_img_path)


if __name__ == '__main__':
    if not os.path.exists(origin_weight_path):
        download_model()
    main()
