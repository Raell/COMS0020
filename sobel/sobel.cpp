#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>

#define PI 3.14159265

using namespace cv;

std::vector<double> rhoValues;
std::vector<double> thetaValues;

Mat convolution(Mat &input, int size, int direction, Mat kernel, cv::Size image_size);

void drawLines(Mat &image, std::vector<double> &rhoValues, std::vector<double> &thetaValues);

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size);

Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size);

void getThresholdedMag(Mat &input, Mat &output);

Mat get_houghSpace(Mat &thresholdMag, Mat &gradientDirection, int width, int height);

void collect_lines_from_houghSpace(Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues,
                                   double threshold);

int main(int argc, const char **argv) {

    const char *imgName = argv[1];

    Mat image;
    image = imread(imgName, 1);

    // namedWindow( "Original Image", CV_WINDOW_AUTOSIZE );
    // imshow( "Original Image", image );

    cvtColor(image, image, CV_BGR2GRAY);


    //init kernels
    Mat dxKernel = (Mat_<double>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);

    Mat dyKernel = (Mat_<double>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1);

    Mat thresholdedMag;
    thresholdedMag.create(image.size(), CV_64F);

    Mat image_clone = imread(imgName, 1);

    Mat dfdx = convolution(image, 3, 0, dxKernel, image.size());
    Mat dfdy = convolution(image, 3, 1, dyKernel, image.size());

    Mat gradientMagnitude = getMagnitude(dfdx, dfdy, image.size());
    Mat gradientDirection = getDirection(dfdx, dfdy, image.size());

    getThresholdedMag(gradientMagnitude, thresholdedMag);

    Mat h = get_houghSpace(thresholdedMag, gradientDirection, image.cols, image.rows);

    //ad-hoc method to find threshold...maybe use hardcoded values?
    double min, max;
    cv::minMaxLoc(h, &min, &max);
    double houghSpaceThreshold = min + ((max - min) / 2);


    //maybe use a specialized class - Line<Pair<Rho, Theta>>?
    std::vector<double> rho;
    std::vector<double> theta;

    collect_lines_from_houghSpace(h, rho, theta, houghSpaceThreshold);

    drawLines(image_clone, rho, theta);

    return 0;
}

void drawLines(Mat &image, std::vector<double> &rhoValues, std::vector<double> &thetaValues) {
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

        line(image, point1, point2, Scalar(0, 255, 0), 2);
    }

    imwrite("result/foundLines.jpg", image);
}

void collect_lines_from_houghSpace(Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues,
                                   double threshold) {
    /*
     * Populates the line vectors and thresholds the houghspace.
     */

    for (int y = 0; y < houghSpace.rows; y++) {
        for (int x = 0; x < houghSpace.cols; x++) {
            double val = houghSpace.at<double>(y, x);

            if (val > threshold) {
                rhoValues.push_back(y);
                thetaValues.push_back(x);
                houghSpace.at<double>(y, x) = 255;
            } else {
                houghSpace.at<double>(y, x) = 0.0;
            }
        }
    }
    imwrite("output/houghSpace.jpg", houghSpace);
}

Mat get_houghSpace(Mat &thresholdMag, Mat &gradientDirection, int width, int height) {

    Mat hough_space;
    hough_space.create(2 * (width + height), 360, CV_64F);
    double angle_range = 20;

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
    imwrite("result/HoughSpace.jpg", hough_space);
    return hough_space;
}

Mat convolution(Mat &input, int size, int direction, Mat kernel, cv::Size image_size) {
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

            double dxVal = 0.0;
            double dyVal = 0.0;
            double gradientVal = 0.0;

            dxVal = dfdx.at<double>(y, x);
            dyVal = dfdy.at<double>(y, x);

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

void getThresholdedMag(Mat &input, Mat &output) {
    Mat img;
    img.create(input.size(), CV_64F);

    normalize(input, img, 0, 255, NORM_MINMAX);

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            double val = 0;
            val = img.at<double>(y, x);

            if (val > 100) output.at<double>(y, x) = 255.0;
            else output.at<double>(y, x) = 0.0;
        }
    }

    imwrite("result/thresholded.jpg", output);
}
