# Helmet_Detection_yolov10_streamlit

## Description
A simple web demo for human detection using YOLOv9 and Streamlit.

## 1. CLONE this repo##
```
git clone https://github.com/ThiDungNguyen/Helmet_Detection_yolov10.git
```
## 2. How to use Using conda environment
1. Create new conda environment and install required dependencies:
```
$ conda create -n yolo_streamlit -y python=3.11
$ conda activate yolo_streamlit
$ pip3 install -r requirements.txt
```
2. Host streamlit app
```
$streamlit run app.py
```


in training dataset, it contains the images and the box location. If no labeled data is available, it must be done before training module. To do it the program called labelImg could be beneficial 
