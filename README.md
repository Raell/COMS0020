### Current Project structure



* _CSVS_ : This folder contains annotated ground truth positions for faces and dartboards. 

* `draw.cpp`: takes an image as argument, looks up its corresponding csv file for ground truths, then draws the ground-truth bounding boxes. The resulting 
image is called `annotated.jpg`

* `face_detector.cpp`: takes an image as argument, uses the viola jones cascade to detect faces in the image. The resulting image is called `face_detected.jpg`
 