This is an basic image tracker which we have made using DeepSort and yolov8 models.
where Yolov8 is used for object detection and deepsort is used for object tracking
where in our code as suitable for my laptop I've used __yolov8s.pt__ where "s" stands for "small",this model provides fast speed with decent accuracy as compared to **yolov8n.pt**.Yolov8n.pt is the nano model which presents very fast speed with low accuracy.
Some other yolov8 models are __yolov8m.pt,yolov8l.pt,yolov8x.pt__ these are the models arranged in increasing accuracy and decreasing speed with **yolov8x.pt** being the SOTA.
and as of the code i would recommend creating virtual enviroment before installing the libraraies and running the code.
Required libraries are:
__ultralytics==8.0.152
deep_sort_realtime==1.3.2
opencv-python==4.8.1.78
numpy==1.26.4__
