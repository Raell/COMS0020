### Current Project structure



* `CSVs/` : contains annotated ground-truth positions for faces and dartboards. 

* `draw.cpp`: takes an image as argument, looks up its corresponding csv file for ground-truths, then draws the  bounding boxes in red. The resulting 
image is then stored in `annotated.jpg`

* `face_detector.cpp`: takes an image as argument, uses the viola jones cascade to detect faces in the image. Detections are drawn with green bounding-boxes.  The resulting image is then stored in `face_detected.jpg`. 
 
 * `frontalface.xml`: Viola Jones Cascade classifier. 
 
 * `negatives/`: non-dartboard images 
 
