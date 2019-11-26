/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String cascade_name = "dartcascade/best.xml";
CascadeClassifier cascade;

/** @function main */
int main(int argc, const char **argv) {
    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    // 3. Detect Faces and Display Result
    detectAndDisplay(frame);

    std::string filename(argv[1]);

    string::size_type i = filename.rfind('.', filename.length());
    if (i != string::npos) {
        filename.replace(i, 4, "");
    }


    // 4. Save Result Image
    imwrite(filename + "detected.jpg", frame);

    return 0;
}

vector <Rect> mergeDetections(vector <Rect> &detected) {
    std::vector <Rect> singles;
    std::vector <Rect> overlaps;
    for (auto rect_1: detected) {
        bool isSingle = true;
        for (auto rect_2: detected) {
            if (rect_1 == rect_2) {
                continue;
            }
            bool intersects = ((rect_1 & rect_2).area() > 0);
            if (intersects) {
                isSingle = false;
                break;
            }
        }
        if (isSingle) {
            singles.push_back(std::move(rect_1));
        } else {
            overlaps.push_back(std::move(rect_1));
        }
    }
    groupRectangles(overlaps, 1, 0.8);

    //we don't want to exclude single boxes
    for (auto r: singles) {
        overlaps.push_back(std::move(r));
    }
    return overlaps;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame) {
    std::vector <Rect> faces;
    Mat frame_gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    //2.5 merge detections
    auto merged = mergeDetections(faces);

    // 3. Print number of Faces found
    std::cout << faces.size() << std::endl;

    // 4. Draw box around faces found
    for (int i = 0; i < merged.size(); i++) {
        rectangle(frame, Point(merged[i].x, merged[i].y),
                  Point(merged[i].x + merged[i].width, merged[i].y + merged[i].height), Scalar(0, 255, 0), 2);
    }

}
