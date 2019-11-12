#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>


using namespace std;
using namespace cv;

vector <Rect> getGroundTruthsFromCSV(string csv);
string get_csv_file(const char *imgName);
vector <Rect> detectAndDisplay(Mat frame);
float f1_test(vector <Rect> &detected, vector <Rect> &actual, float threshold);


String CASCADE_NAME = "frontalface.xml";
CascadeClassifier cascade;


int main(int argc, const char **argv) {

    const char *imgName = argv[1];

    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(CASCADE_NAME)) {
        printf("--(!)Error loading Cascade\n");
        return -1;
    };

    string csv_file_path = get_csv_file(imgName);

    vector <Rect> ground_truths = getGroundTruthsFromCSV(csv_file_path);

    vector <Rect> detected = detectAndDisplay(frame);

    f1_test(detected, ground_truths, 50.0);

    imwrite("face_detected.jpg", frame);
}

string get_csv_file(const char *imgName) {

    string fileExtension = "points.csv";
    string filePrefix = "CSVs/faces/";
    string current_line;

    std::string csv_filename(imgName);
    string::size_type i = csv_filename.rfind('.', csv_filename.length());
    if (i != string::npos) {
        csv_filename.replace(i, fileExtension.length(), fileExtension);
    }
    return filePrefix + csv_filename;
}


vector <Rect> getGroundTruthsFromCSV(string csv) {
    const char *c = csv.c_str();
    vector <Rect> ground_truths;

    string current_line;
    ifstream inputFile(c);

    if (inputFile.peek() == std::ifstream::traits_type::eof()) {
        std::cout << "No CSV file found. F1 score cannot be calculated" << '\n';
        exit(EXIT_FAILURE);

    }


    while (getline(inputFile, current_line)) {

        std::vector<int> values;

        std::stringstream convertor(current_line);
        std::string token;

        while (std::getline(convertor, token, ',')) {
            values.push_back(std::atoi(token.c_str()));
        }

        ground_truths.push_back(Rect(values[0], values[1], values[2], values[3]));
    }

    return ground_truths;

}

float f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, float threshold) {
    int truePositives = 0;
    int falsePositives = 0;

    for (int i = 0; i < detected.size(); i++) {
        bool matchFound = false;
        for (int j = 0; j < ground_truth.size(); j++) {
            Rect intersection = detected[i] & ground_truth[j];
            float intersectionArea = intersection.area();

            if (intersectionArea > 0) {
                float matchPercentage = (intersectionArea / ground_truth[j].area()) * 100;
                if (matchPercentage > threshold) {
                    truePositives++;
                    matchFound = true;
                    break;
                }
            }
        }
        if (!matchFound) {
            falsePositives++;
        }
    }

    std::cout << "true positives: " << truePositives << ", false positives: " << falsePositives << "\n";

    // Precision = TP / (TP + FP)
    // Recall = TPR (True Positive Rate)
    // F1 = 2((PRE * REC)/(PRE + REC))

    float precision = (float) truePositives / ((float) truePositives + (float) falsePositives);
    float recall = (float) truePositives / (float) ground_truth.size();

    float f1 = 2 * ((precision * recall) / (precision + recall));

    cout << "f1 score: " << f1 << "\n";

    return f1;
}

vector <Rect> detectAndDisplay(Mat frame) {
    Mat frame_gray;
    vector <Rect> detected;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, detected, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    // 3. Print number of Faces found
    cout << "faces detected: " << detected.size() << std::endl;

    // 4. Draw box around faces found
    for (int i = 0; i < detected.size(); i++) {
        rectangle(frame, Point(detected[i].x, detected[i].y),
                  Point(detected[i].x + detected[i].width, detected[i].y + detected[i].height),
                  Scalar(0, 255, 0), 2);
    }
    return detected;
}





