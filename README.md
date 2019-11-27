### Current Project structure



* `CSVs/` : contains annotated ground-truth positions for faces and dartboards. 

* `draw.cpp`: takes an image as argument, looks up its corresponding csv file for ground-truths, then draws the  bounding boxes in red. The resulting 
image is then stored in `annotated.jpg`

* `face_detector.cpp`: takes an image as argument, uses the viola jones cascade to detect faces in the image. Detections are drawn with green bounding-boxes.  The resulting image is then stored in `face_detected.jpg`. 
 
 * `frontalface.xml`: Viola Jones Cascade classifier. 
 
 * `negatives/`: non-dartboard images 
 
For detection results compile using:

`g++ face.cpp -o face -std=c++11 -I/usr/include/opencv -I/usr/include/opencv2 -lopencv_calib3d -lopencv_imgproc -lopencv_contrib -lopencv_legacy -lopencv_core -lopencv_ml -lopencv_features2d -lopencv_objdetect -lopencv_flann -lopencv_video -lopencv_highgui`

```
Usage: ./face [-i input_folder] [-c cascade_xml] [-o output_folder] [-a annotations_folder] [-m]

Options:
-i input_folder, specifies image folder
-c cascade_xml, specifies trained cascade.xml file to use
-o output_folder, specifies output folder for results
-a annotations_folder, specifies folder containing annotations in csv files
-m, flag to set if output boxes of Viola-Jones to be merged
```


This runs the cascade specifed on all i ∈ [0,15], dart{i}.jpg images in the input folder with ground truth from the annotations folder and returns the results in the ouput folder. `-m` flag determines if the bounding boxes should be merged after Viola-Jones detector step.
