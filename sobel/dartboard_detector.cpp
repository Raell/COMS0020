#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <array>

#define PI 3.14159265

#define GRADIENT_THRESHOLD 200
#define ANGLE_RANGE 20

using namespace cv;
using namespace std;

class Line {
public:
    int rho;
    int theta;

    Line() : rho(0), theta(0) {}

    Line(int rho, int theta) {
        this->rho = rho;
        this->theta = theta;
    }
};

class Circle {
public:
    int x;
    int y;
    int r;

    Circle() : x(0), y(0), r(0) {}

    Circle(int x, int y, int r) {
        this->x = x;
        this->y = y;
        this->r = r;
    }

    double area() {
        return PI * pow(r, 2);
    }
};

std::ostream &operator<<(std::ostream &strm, const Circle &circle) {
    return strm << "Circle: x= " << circle.x << " y=" << circle.y << " r=" << circle.r;
}

String CASCADE_NAME = "../dartcascade/best.xml";
CascadeClassifier cascade;

vector <Rect> getGroundTruthsFromCSV(string csv);
string get_csv_file(const char *imgName);
vector <Rect> detectAndDisplay(Mat frame);
float f1_test(vector <Rect> &detected, vector <Rect> &actual, float threshold);

Mat convolution(Mat &input, int direction, Mat kernel, cv::Size image_size);

void drawLines(Mat &image, Mat thresholdedMag, std::vector <Line> &detected_lines);
void drawCircles(Mat &image, vector <Circle> &circles);

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size);
Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size);
void getThresholdedMag(Mat &input, Mat &output, double gradient_threshold);
Mat get_houghSpaceLines(Mat &thresholdMag, Mat &gradientDirection, int width, int height);

vector <Line> collect_lines_from_houghSpace(Mat &houghSpace, double threshold);

double calculate_houghLines_voting_threshold(Mat &hough_space);
double calculate_houghCircles_voting_threshold(std::vector < std::vector < std::vector < int >> > &hough_space);

double calculate_houghCircles_voting_threshold(std::vector < std::vector < std::vector < int >> > &hough_space) {
    return 20;
}

void pipeline(Mat &frame);

double calculate_houghLines_voting_threshold(Mat &hough_space) {
    double max, min;
    cv::minMaxLoc(hough_space, &min, &max);
    double houghSpaceThreshold = min + ((max - min) / 2);
    return houghSpaceThreshold;
}

void drawCircles(Mat &image, vector <Circle> &circles) {
    for (auto &c: circles) {
        Point center(c.x, c.y);
        cv::circle(image, center, c.r, Scalar(0, 0, 255), 2, 8, 0);
    }

    imwrite("result/foundCircles.jpg", image);
}

vector <Circle> houghCircles(Mat &image, Mat &thresholdMag, Mat &gradient_dir, int voting_threshold) {

    int radius = image.rows / 2;

    int rows{image.rows};
    int cols{image.cols};
    int initialValue{0};


    // Define 3 dimensional vector and initialize it.
    //create a houghspace parameterized on circle centre (x,y) and radius (r)

    std::vector < std::vector < std::vector < int >> >
    houghSpace(rows, std::vector < std::vector < int >> (cols, std::vector<int>(radius, initialValue)));

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {


            /* Results seem to be better without the thresholding filter...

              if (thresholdMag.at<double>(y, x) == 0) {
                continue;
            }*/


            for (int r = 0; r < radius; r++) {
                int x0 = x - (int) (r * cos(gradient_dir.at<double>(y, x)));
                int y0 = y - (int) (r * sin(gradient_dir.at<double>(y, x)));

                //make sure the centre lies within the image
                if (x0 >= 0 && x0 < image.cols && y0 >= 0 && y0 < image.rows) {
                    houghSpace[y0][x0][r]++;
                }

                x0 = x + (int) (r * cos(gradient_dir.at<double>(y, x)));
                y0 = y + (int) (r * sin(gradient_dir.at<double>(y, x)));
                if (x0 >= 0 && x0 < image.cols && y0 >= 0 && y0 < image.rows) {
                    houghSpace[y0][x0][r]++;
                }

            }
        }
    }
    vector <Circle> circles;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int r = 0; r < radius; r++) {
                if (houghSpace[y][x][r] > voting_threshold) {
                    Circle c = Circle(x, y, r);
                    circles.push_back(c);
                }
            }
        }
    }

    return circles;
}

void pipeline(Mat &frame) {

    Mat dxKernel = (Mat_<double>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);

    Mat dyKernel = (Mat_<double>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1);


    //get viola-jones detections and draw then in GREEN.
    auto violaJonesDetections = detectAndDisplay(frame);

    imwrite("result/violaJonesDetections.jpg", frame);

    //for every detection, apply hough transforms to find the number of circles and lines
    for (auto &rect: violaJonesDetections) {

        //expand the rectangle to retain contextual info around the edges
        cv::Point inflationPoint(-20, -20);
        cv::Size inflationSize(20, 20);
        rect += inflationPoint;
        rect += inflationSize;

        auto rgb_viola_jones = frame(rect);
        Mat gray_viola_jones;
        cvtColor(rgb_viola_jones, gray_viola_jones, CV_BGR2GRAY);

        Mat dfdx = convolution(gray_viola_jones, 0, dxKernel, gray_viola_jones.size());
        Mat dfdy = convolution(gray_viola_jones, 1, dyKernel, gray_viola_jones.size());

        Mat gradientMagnitude = getMagnitude(dfdx, dfdy, gray_viola_jones.size());
        Mat gradientDirection = getDirection(dfdx, dfdy, gray_viola_jones.size());

        Mat thresholdedMag;
        thresholdedMag.create(gray_viola_jones.size(), CV_64F);
        getThresholdedMag(gradientMagnitude, thresholdedMag, GRADIENT_THRESHOLD);

        auto circles = houghCircles(gray_viola_jones, thresholdedMag, gradientDirection, 25);
        drawCircles(rgb_viola_jones, circles);
        cout << "circles detected " << circles.size() << std::endl;

        auto houghSpace = get_houghSpaceLines(thresholdedMag, gradientDirection, gray_viola_jones.cols,
                                              gray_viola_jones.rows);
        auto houghLinesThreshold = calculate_houghLines_voting_threshold(houghSpace);
        auto lines = collect_lines_from_houghSpace(houghSpace, houghLinesThreshold);
        cout << "lines detected " << lines.size() << std::endl;
        drawLines(rgb_viola_jones, thresholdedMag, lines);


        //TODO: write a heuristic algorithm (based on the positions of the circles and lines)
        // to determine the existence of a dartboard  within the current bounding box

        cout << "######################" << std::endl;
    }

    //TODO: Load ground truths and keep track of TP, FP etc to compute F1-score.

    imwrite("result/detections.jpg", frame);
}

int main(int argc, const char **argv) {

    const char *imgName = argv[1];

    Mat image;
    image = imread(imgName, 1);

    pipeline(image);
    return 0;
}

void drawLines(Mat &image, Mat thresholdedMag, std::vector <Line> &detected_lines) {

    Mat lines(image.size(), image.type(), Scalar(0));
    int width = image.cols;
    int height = image.rows;
    int centreX = image.cols / 2;
    int centreY = image.rows / 2;
    for (int i = 0; i < detected_lines.size(); i++) {

        Point point1, point2;
        double theta = detected_lines[i].theta;
        double rho = detected_lines[i].rho;

        double radians = theta * (PI / 180);

        //std::cout << rho << "and" << radians << '\n';

        double a = cos(radians);
        double b = sin(radians);
        double x0 = a * (rho - width - height);
        double y0 = b * (rho - width - height);

        point1.x = cvRound(x0 + 1000 * (-b));
        point1.y = cvRound(y0 + 1000 * (a));
        point2.x = cvRound(x0 - 1000 * (-b));
        point2.y = cvRound(y0 - 1000 * (a));

        line(lines, point1, point2, Scalar(0, 0, 255), 1);
        line(image, point1, point2, Scalar(0, 0, 255), 1);

    }

    thresholdedMag.convertTo(thresholdedMag, CV_8U);

    Mat overlay;
    overlay.zeros(image.size(), image.type());
    lines.copyTo(overlay, thresholdedMag);

    imwrite("result/lines.jpg", lines);
    imwrite("result/overlay.jpg", overlay);
    imwrite("result/foundLines.jpg", image);
}

vector <Line> collect_lines_from_houghSpace(Mat &houghSpace,
                                            double threshold) {
    /*
     * Populates the line vector & thresholds the houghspace.
     */
    std::vector <Line> lines;

    for (int y = 0; y < houghSpace.rows; y++) {
        for (int x = 0; x < houghSpace.cols; x++) {
            double val = houghSpace.at<double>(y, x);

            if (val > threshold) {
                Line l = Line(y, x);
                lines.push_back(l);
                houghSpace.at<double>(y, x) = 255;
            } else {
                houghSpace.at<double>(y, x) = 0.0;
            }
        }
    }
    imwrite("result/houghSpace.jpg", houghSpace);
    return lines;
}

Mat get_houghSpaceLines(Mat &thresholdMag, Mat &gradientDirection, int width, int height) {

    Mat hough_space;
    hough_space.create(2 * (width + height), 360, CV_64F);
    double angle_range = ANGLE_RANGE;

    for (int y = 0; y < thresholdMag.rows; y++) {
        for (int x = 0; x < thresholdMag.cols; x++) {
            double value = thresholdMag.at<double>(y, x);
            if (value == 255.0) {
                double direction = gradientDirection.at<double>(y, x);
                double direction_angle;
                if (direction > 0) {
                    direction_angle = (direction * (180 / PI));
                } else {
                    direction_angle = 360 + direction * (180 / PI);
                }

                direction_angle = round(direction_angle);

                //loop from smallest..biggest to allow room for errors
                double smallest = direction_angle - angle_range;
                double biggest = direction_angle + angle_range;

                for (int angle = smallest; angle < biggest; angle++) {
                    double radians = angle * (PI / 180);
                    double rho = (x * cos(radians)) + (y * sin(radians)) + width + height;
                    hough_space.at<double>(rho, angle)++;

                }
            }
        }
    }
    normalize(hough_space, hough_space, 0, 255, NORM_MINMAX);

    return hough_space;
}

Mat convolution(Mat &input, int direction, Mat kernel, cv::Size image_size) {
    Mat output;
    output.create(image_size, CV_64F);

    int kernelRadiusX = (kernel.size[0] - 1) / 2;
    int kernelRadiusY = (kernel.size[1] - 1) / 2;

    // Create padded version of input
    Mat paddedInput;
    copyMakeBorder(input, paddedInput,
                   kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                   BORDER_REPLICATE);

    GaussianBlur(paddedInput, paddedInput, Size(3, 3), 0, 0, BORDER_DEFAULT);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {

            double sum = 0.0;

            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {

                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = (int) paddedInput.at<uchar>(imagex, imagey);
                    double kernalval = kernel.at<double>(kernelx, kernely);

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            output.at<double>(i, j) = sum;
        }
    }

    Mat img;
    img.create(input.size(), CV_64F);
    // Normalise to avoid out of range and negative values
    normalize(output, img, 0, 255, NORM_MINMAX);

    //Save thresholded image
    if (direction == 0) imwrite("result/dfdx.jpg", img);
    else imwrite("result/dfdy.jpg", img);
    return output;
}

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size) {
    Mat output;
    output.create(image_size, CV_64F);

    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {

            double dxVal = 0.0;
            double dyVal = 0.0;
            double magnitudeVal = 0.0;

            dxVal = dfdx.at<double>(y, x);
            dyVal = dfdy.at<double>(y, x);

            magnitudeVal = sqrt(pow(dxVal, 2) + pow(dyVal, 2));

            output.at<double>(y, x) = magnitudeVal;
        }
    }

    Mat img;
    img.create(dfdx.size(), CV_64F);

    normalize(output, img, 0, 255, NORM_MINMAX);

    imwrite("result/magnitude.jpg", img);
    return output;
}

Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size) {

    Mat output;
    output.create(image_size, CV_64F);

    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {

            double gradientVal = 0.0;

            double dxVal = dfdx.at<double>(y, x);
            double dyVal = dfdy.at<double>(y, x);

            // Calculate direction
            if (dxVal != 0 && dyVal != 0) gradientVal = atan2(dyVal, dxVal);
            else gradientVal = (double) atan(0);

            output.at<double>(y, x) = gradientVal;

        }
    }

    Mat img;
    img.create(dfdx.size(), CV_64F);

    normalize(output, img, 0, 255, NORM_MINMAX);

    imwrite("result/direction.jpg", img);
    return output;
}

void getThresholdedMag(Mat &input, Mat &output, double gradient_threshold) {
    Mat img;
    img.create(input.size(), CV_64F);

    normalize(input, img, 0, 255, NORM_MINMAX);

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            double val = 0;
            val = img.at<double>(y, x);

            if (val > gradient_threshold) output.at<double>(y, x) = 255.0;
            else output.at<double>(y, x) = 0.0;
        }
    }

    imwrite("result/thresholded.jpg", output);
}

vector <Rect> detectAndDisplay(Mat frame) {
    Mat frame_gray;
    vector <Rect> detected;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);


    //load the cascade
    if (!cascade.load(CASCADE_NAME)) {
        printf("--(!)Error loading Cascade\n");
        std::exit(0);
    };


    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, detected, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    // 2.5 Merge overlapping rectangles
    groupRectangles(detected, 1, 0.8);


    // 3. Print number of Faces found
    cout << "dartboards detected: " << detected.size() << std::endl;

    // 4. Draw box around faces found
    for (int i = 0; i < detected.size(); i++) {
        rectangle(frame, Point(detected[i].x, detected[i].y),
                  Point(detected[i].x + detected[i].width, detected[i].y + detected[i].height),
                  Scalar(0, 255, 0), 2);
    }
    return detected;
}

float f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, float threshold) {
    int truePositives = 0;
    int falsePositives = 0;
    for (int i = 0; i < detected.size(); i++) {
        bool matchFound = false;
        for (int j = 0; j < ground_truth.size(); j++) {
            Rect intersection = detected[i] & ground_truth[j];
            Rect box_union = detected[i] | ground_truth[j];
            float intersectionArea = intersection.area();
            float unionArea = box_union.area();
            if (intersectionArea > 0) {
                float matchPercentage = (intersectionArea / unionArea) * 100;
                if (matchPercentage > threshold) {
                    truePositives++;
                    matchFound = true;
                    cout << intersectionArea << endl;
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
    int falseNegatives = ground_truth.size() - truePositives;
    float precision = (float) truePositives / ((float) truePositives + (float) falsePositives);
    float recall = (float) truePositives / (float) ground_truth.size();
    float f1;
    if (precision > 0 && recall > 0) {
        f1 = 2 * (precision * recall) / (precision + recall);
    } else if (!truePositives && !falsePositives && !falseNegatives) {
        f1 = 1;
        recall = 1;
    } else {
        f1 = 0;
        recall = 0;
    }
    return f1;
}