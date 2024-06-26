git clone https://github.com/THU-MIG/yolov10.git

# Commented out IPython magic to ensure Python compatibility.
# %cd yolov10
pip install -q -r requirements.txt
pip install -e .
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
pwd

from ultralytics import YOLOv10

MODEL_PATH = 'yolov10n.pt'
model = YOLOv10 (MODEL_PATH)

!gdown '1tr9PSRRdlC2pNir7jsYugpSMG-7v32VJ' -O './images/'

IMG_PATH = './images/HCMC_Street.jpg'
result = model(source = IMG_PATH )[0]

result.save('./images/HCMC_Street_predict.png')