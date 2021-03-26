
from darknet import *
import cv2

network, class_names, colors = load_network("cfg/yolov4.cfg","cfg/coco.data","yolov4.weights")
img = load_image("test.jpg".encode("ascii"), 0, 0)
detections = detect_image(network,class_names,img)
print_detections(detections)	
image = cv2.imread("test.jpg")
image = draw_boxes(detections, image, colors)
cv2.imwrite("frame.jpg",image)
