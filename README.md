### Current Project structure



* `CSVs/` : contains annotated ground-truth positions for faces and dartboards. 

* `annotated/` : contains result files. 

* `dartcascade/` : contains dartboard classfier xml for Viola-Jones. 

* `face_detector.cpp`: takes an image as argument, uses the Viola-Jones cascade to detect faces in the image. Detections are drawn with green bounding-boxes and ground truths with red bounding-boxes.
 
* `frontalface.xml`: Viola Jones face Cascade classifier. 
 
* `negatives/`: non-dartboard images 
 
* `ipcv-coursework.pdf`: project writeup reflecting on methodology and results ([link](ipcv-coursework.pdf))
 
For detection results compile using:

`g++ face_detector.cpp -o face_detector -std=c++11 -I/usr/include/opencv -I/usr/include/opencv2 -lopencv_calib3d -lopencv_imgproc -lopencv_contrib -lopencv_legacy -lopencv_core -lopencv_ml -lopencv_features2d -lopencv_objdetect -lopencv_flann -lopencv_video -lopencv_highgui`

```
Usage: ./face_detector [-i input_folder] [-d image_file] [-c cascade_xml] [-o output_folder] [-a annotations_folder] [-p PIPELINE] [-s SHAPES]
Options:
-i input_folder, specifies image folder
-c cascade_xml, specifies trained cascade.xml file to use
-o output_folder, specifies output folder for results
-a annotations_folder, specifies folder containing annotations in csv files
-d image_file, run single dart image number from folder
-p PIPELINE, flag to set what pipeline is run
  Parameters:
  FULL  - entire detection pipeline with merge and canny edges (default) (pg 4 results)
  CANNY - detection pipeline with canny edges
  MERGE - detection pipeline with merge
  BASIC - basic detection pipeline (pg 3 results)
  VJ    - only use Viola-Jones detector (pg 2 results)
-s SHAPES, flag to show detected shapes
  Parameters:
  LINES - show detected lines
  CIRCLES - show detected circles
  ELLIPSES - show detected ellipses
```


This runs the cascade specifed on all i ∈ [0,15], dart{i}.jpg images in the input folder with ground truth from the annotations folder and returns the results in the ouput folder.
