#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>

#define PI 3.14159265

#define GRADIENT_THRESHOLD 200.0
#define ANGLE_RANGE 20

using namespace cv;
using namespace std;

Mat convolution(Mat &input, int direction, Mat kernel, cv::Size image_size);

void drawLines(Mat &image, Mat thresholdedMag, std::vector<double> &rhoValues, std::vector<double> &thetaValues);

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size);

Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size);

void getThresholdedMag(Mat &input, Mat &output, double gradient_threshold);

Mat get_houghSpace(Mat &thresholdMag, Mat &gradientDirection, int width, int height);

void collect_lines_from_houghSpace(Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues,
                                   double threshold);
vector<tuple<Point, double, double, double>> houghEllipse(Mat &thresholdMag, int width, int height, double min_major, double min_minor, int detection_threshold);

void drawEllipse(Mat &image, Mat thresholdedMag, vector<tuple<Point, double, double, double>> hough_ellipse);

double calculate_houghSpace_voting_threshold(Mat &hough_space) {
    double max, min;
    cv::minMaxLoc(hough_space, &min, &max);
    double houghSpaceThreshold = min + ((max - min) / 2);
    return houghSpaceThreshold;

}

int main(int argc, const char **argv) {

    const char *imgName = argv[1];

    Mat image;
    image = imread(imgName, 1);


    cvtColor(image, image, CV_BGR2GRAY);


    //init kernels
    Mat dxKernel = (Mat_<double>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);

    Mat dyKernel = (Mat_<double>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1);

    Mat image_clone = imread(imgName, 1);

    Mat dfdx = convolution(image, 0, dxKernel, image.size());
    Mat dfdy = convolution(image, 1, dyKernel, image.size());

    Mat gradientMagnitude = getMagnitude(dfdx, dfdy, image.size());
    Mat gradientDirection = getDirection(dfdx, dfdy, image.size());

    Mat thresholdedMag;
    thresholdedMag.create(image.size(), CV_64F);

    getThresholdedMag(gradientMagnitude, thresholdedMag, GRADIENT_THRESHOLD);

    Mat houghSpace = get_houghSpace(thresholdedMag, gradientDirection, image.cols, image.rows);

    double houghSpaceThreshold = calculate_houghSpace_voting_threshold(houghSpace);

    std::vector<double> rho;
    std::vector<double> theta;

    collect_lines_from_houghSpace(houghSpace, rho, theta, houghSpaceThreshold);

    drawLines(image_clone, thresholdedMag, rho, theta);

    double min_major = 70;
    double min_minor = 50;
    int detection_threshold = 100;

    // vector<tuple<Point, double, double, double>> hough_ellipse = houghEllipse(thresholdedMag, image.cols, image.rows,
    //     min_major, min_minor, detection_threshold);


    drawEllipse(image_clone, thresholdedMag, hough_ellipse);

    return 0;
}

void drawEllipse(Mat &image, Mat thresholdedMag, 
    vector<tuple<Point, double, double, double>> hough_ellipse) {

    Mat ellipses(image.size(), image.type(), Scalar(0));

    for (int i = 0; i < hough_ellipse.size(); i++) {

        Point center = get<0>(hough_ellipse[i]);
        double major_axis = get<1>(hough_ellipse[i]);
        double minor_axis = get<2>(hough_ellipse[i]);
        double alpha = get<3>(hough_ellipse[i]);
        Size axes(major_axis, minor_axis);

        ellipse(ellipses, center, axes, alpha, 0, 360,
            Scalar(0, 0, 255),
            2);
        ellipse(image, center, axes, alpha, 0, 360,
            Scalar(0, 0, 255),
            2);
    }

    thresholdedMag.convertTo(thresholdedMag, CV_8U);

    Mat overlay;
    overlay.zeros(image.size(), image.type());
    ellipses.copyTo(overlay, thresholdedMag);

    imwrite("result/ellipses.jpg", ellipses);
    imwrite("result/overlay.jpg", overlay);
    imwrite("result/foundEllipses.jpg", image);

}

void drawLines(Mat &image, Mat thresholdedMag, std::vector<double> &rhoValues, std::vector<double> &thetaValues) {

    Mat lines(image.size(), image.type(), Scalar(0));
    int width = image.cols;
    int height = image.rows;
    int centreX = image.cols / 2;
    int centreY = image.rows / 2;

    for (int i = 0; i < rhoValues.size(); i++) {

        Point point1, point2;
        double theta = thetaValues[i];
        double rho = rhoValues[i];

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

void collect_lines_from_houghSpace(Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues, double threshold) {
    /*
     * Populates the line vectors, and thresholds the houghspace.
     */

    for (int y = 0; y < houghSpace.rows; y++) {
        for (int x = 0; x < houghSpace.cols; x++) {
            double val = houghSpace.at<double>(y, x);

            if (val > threshold) {
                rhoValues.push_back(y);
                thetaValues.push_back(x);
                //   std::cout << x << " ";
                houghSpace.at<double>(y, x) = 255;
            } else {
                houghSpace.at<double>(y, x) = 0.0;
            }
        }
    }
    imwrite("result/houghSpace.jpg", houghSpace);
}

Mat get_houghSpace(Mat &thresholdMag, Mat &gradientDirection, int width, int height) {

    Mat hough_space;
    hough_space.create(2 * (width + height), 360, CV_64F);
    double angle_range = ANGLE_RANGE;

    for (int y = 0; y < thresholdMag.rows; y++) {
        for (int x = 0; x < thresholdMag.cols; x++) {
            double value = thresholdMag.at<double>(y, x);
            if (value > 250) {
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

vector<tuple<Point, double, double, double>> houghEllipse(Mat &thresholdMag, int width, int height, double min_major, double min_minor, int detection_threshold) {
    // Hough Ellipse detection based on 
    // Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection method." Pattern Recognition, 2002. Proceedings. 16th International Conference on. Vol. 2. IEEE, 2002

    // Find all edges coordinates by using the thresholded magnitude
    vector<Point> locations;    
    Mat mag = thresholdMag.clone();
    mag.convertTo(mag, CV_8U);
    findNonZero(mag, locations);

    // Stores found ecllipse in (x0, y0, a, b, alpha) format
    vector<tuple<Point, double, double, double>> ellipse;

    // Iterate on all edge coordinates 
    for (int m = 0; m < locations.size() - 2; m++) {
        
        // Get first pixel to lookup
        int x1 = locations[m].x;
        int y1 = locations[m].y;

        for (int n = m + 1; n < locations.size() - 1; n++) {

            // Get second pixel to lookup
            int x2 = locations[n].x;
            int y2 = locations[n].y;

            double major_axis = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
            if (major_axis < min_major) continue;

            // Get half-length of major axis (a) and orientation (alpha)
            double a = major_axis/2;
            double alpha;
            double dy = y2 - y1;
            double dx = x2 - x1;

            if (dx != 0 && dy != 0) alpha = atan2(dy, dx);
            else alpha = (double) atan(0);

            // Calculate center of ellipse
            int x0 = cvRound((x1 + x2)/2);
            int y0 = cvRound((y1 + y2)/2);


            Mat accumulator(1,
                (int)(sqrt(pow(width,2) + pow(height,2)) + 1), 
                CV_64F);

            for (int o = n + 1; o < locations.size(); o++) {

                // Get third pixel to lookup
                int x = locations[o].x;
                int y = locations[o].y;

                double d = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
                if (d < min_minor) continue;
                if (d > a) continue;

                // Get half-length of minor-axis (b)
                double f_square = pow(x - x2, 2) + pow(y - y2, 2);
                double cos_tau_square = pow(
                    (pow(a,2) + pow(d,2) - f_square)/(2*a*d)
                    ,2);

                // Assume b > 0 and avoid division by 0
                double k = pow(a,2) - pow(d,2) * cos_tau_square;
                if (k > 0 && cos_tau_square < 1) {
                    int b = cvRound(sqrt((pow(a,2) * pow(d,2) * (1 - cos_tau_square))/k));
                    accumulator.at<double>(0, b)++;
                }
                
            }

            double max;
            Point maxLoc;

            // Adds ellipse if the maximum of the accumulator exceeds threshold
            minMaxLoc(accumulator, NULL, &max, NULL, &maxLoc);
            if (max > detection_threshold) {
                tuple<Point,double,double,double> params (Point(x0, y0), a, maxLoc.y, alpha);
                ellipse.push_back(params);
            }

        }
    }



    return ellipse;
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
