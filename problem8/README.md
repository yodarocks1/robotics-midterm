# (a) SSD
Using Single-Shot Detector (SSD) in [ssd.py](ssd.py) and the [grocerystore](/grocerystore/) images outputted into [ssd](/ssd/)
SSD-Examples                                 | SSD-Examples 
:-------------------------------------------:|:----------------------:
![](ssd/orlando.jpg)                         | ![](ssd/003_Albany_Grocery_Store_Interior.jpg)
![](ssd/supermercado22.jpg)                  | ![](ssd/106598451_eba244869f.jpg)

# (b) YOLO v8 & YOLO v5 (Both Ultralytics)
YOLO v8                                                |  YOLO v5
:-----------------------------------------------------:|:----------------------:
![](yolov8/1ng10a.jpg)                                 | ![](yolov5/1ng10a.jpg)
![](yolov8/2fgr_supermarkt.jpg)                        | ![](yolov5/2fgr_supermarkt.jpg)
![](yolov8/003_Albany_Grocery_Store_Interior.jpg)      | ![](yolov5/003_Albany_Grocery_Store_Interior.jpg)
![](yolov8/167613_3.jpg)                               | ![](yolov5/167613_3.jpg)
![](yolov8/106598451_eba244869f.jpg)                   | ![](yolov5/106598451_eba244869f.jpg)
![](yolov8/4734.jpg)                                   | ![](yolov5/4734.jpg)
![](yolov8/supermercado22.jpg)                         | ![](yolov5/supermercado22.jpg)
![](yolov8/quake2.jpg)                                 | ![](yolov5/quake2.jpg)
![](yolov8/orlando.jpg)                                | ![](yolov5/orlando.jpg)

# (c) YOLOv8n Desktop vs Pi

## Desktop (32GB RAM, i9-10900KF, RTX 3080)
Average FPS: 85.24

Average CPU Usage: 13.86%

Average GPU Usage: 21.02%

![](desktop.png)

## Pi 4B 4GB RAM
Average FPS: 0.93

Average CPU Usage: 84.31%

![](pi.png)

# (d) Mask R-CNN vs Faster R-CNN vs RetinaNet

[Colab Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)

### Mask R-CNN
![](maskrcnn.png)

### Faster R-CNN
![](fasterrcnn.png)

### RetinaNet
![](retinanet.png)
