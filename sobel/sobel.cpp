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

using Ellipse = tuple<Point, double, double, double, int>;

Mat convolution(Mat &input, int direction, Mat kernel, cv::Size image_size);

void drawLines(Mat &image, Mat thresholdedMag, std::vector<double> &rhoValues, std::vector<double> &thetaValues);

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size);

Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size);

void getThresholdedMag(Mat &input, Mat &output, double gradient_threshold);

Mat get_houghSpace(Mat &thresholdMag, Mat &gradientDirection, int width, int height);

void collect_lines_from_houghSpace(Mat &houghSpace, std::vector<double> &rhoValues, std::vector<double> &thetaValues, double threshold);

vector<Ellipse> houghEllipse(Mat &thresholdMag, int width, int height, tuple<vector<int>, vector<double>> ellipses_thresholds);

void drawEllipse(Mat &image, Mat thresholdedMag, vector<Ellipse> hough_ellipse);

double weighted_params(double prev, double current, int prev_score, int curr_score, bool angles);

Ellipse merge_ellipses(vector<Ellipse, int> &accumulator, Ellipse new_ellipse,  
    double center_distance_threshold, double semimajor_axis_threshold, 
    double semiminor_axis_threshold, double angle_threshold
    );

tuple<vector<int>, vector<double>> calculate_ellipse_detection_threshold(Mat &image, Mat &mag, Mat &dir);


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

    tuple<vector<int>, vector<double>> ellipses_thresholds =  calculate_ellipse_detection_threshold(image, thresholdedMag, gradientDirection);

    vector<Ellipse> hough_ellipse = houghEllipse(thresholdedMag, image.cols, image.rows,
        ellipses_thresholds);

    drawEllipse(image_clone, thresholdedMag, hough_ellipse);

    return 0;
}

double weighted_params(double prev, double current, int prev_score, int curr_score, bool angles) 
{

    if (!angles) {
        return ((prev * prev_score) + (current * curr_score))/(prev_score + curr_score);
    }

    else {
        double curr_rad = current * CV_PI/180;
        double prev_rad = prev * CV_PI/180;
        double avg_sin = prev_score * sin(prev_rad) + curr_score * sin(curr_rad);
        double avg_cos = prev_score * cos(prev_rad) + curr_score * cos(curr_rad);
        return ((avg_sin != 0 && avg_cos != 0) ? atan2(avg_sin, avg_cos) : (double) atan(0));
    }
    
    
}

Ellipse merge_ellipses(vector<Ellipse> &accumulator, Ellipse new_ellipse, 
    double center_distance_threshold, double semimajor_axis_threshold,
    double semiminor_axis_threshold, double angle_threshold) 
{
    
    Point curr_center = get<0>(new_ellipse);
    double curr_maj = get<1>(new_ellipse);
    double curr_min = get<2>(new_ellipse);
    double curr_angle =  get<3>(new_ellipse);
    int curr_score = get<4>(new_ellipse);

    bool found_match = false;

    int i;

    for (i = 0; i < accumulator.size(); i++)
    {
        // Compare the new ellipse to previously found ellipses
        Ellipse prev_ellipse = accumulator[i];

        Point prev_center = get<0>(prev_ellipse);
        double prev_maj = get<1>(prev_ellipse);
        double prev_min = get<2>(prev_ellipse);
        double prev_angle =  get<3>(prev_ellipse);
        int prev_score = get<4>(prev_ellipse);

        double center_dist = sqrt(pow(curr_center.x - prev_center.x, 2) + pow(curr_center.y - prev_center.y, 2));
        double majax_dist = abs(prev_maj - curr_maj);
        double minax_dist = abs(prev_min - curr_min);
        double angle_dist = abs(prev_angle - curr_angle);

        if (angle_dist > 180) 
        {
            angle_dist = 360 - angle_dist;
        }

        if (center_dist < center_distance_threshold && majax_dist < semimajor_axis_threshold && minax_dist < semiminor_axis_threshold && angle_dist < angle_threshold)
        {
            Ellipse weighted_ellipse = make_tuple(
                Point(
                    weighted_params(prev_center.x, curr_center.x, prev_score, curr_score, false),
                    weighted_params(prev_center.y, curr_center.y, prev_score, curr_score, false)
                    ),
                weighted_params(prev_maj, curr_maj, prev_score, curr_score, false),
                weighted_params(prev_min, curr_min, prev_score, curr_score, false),
                weighted_params(prev_angle, curr_angle, 
                    prev_score, curr_score, false),
                prev_score + curr_score
                );

            accumulator[i] = weighted_ellipse;
            return weighted_ellipse;
        }

    }

    if (!found_match) 
    {
        accumulator.push_back(new_ellipse);
        return new_ellipse;
    }

}


tuple<vector<int>, vector<double>> calculate_ellipse_detection_threshold(Mat &image, Mat &mag, Mat &dir) {

    // Thresholds for detection, iteration and quantisation
    int detection_threshold = 200;
    int major_pair_limit = 2;
    int accuracy = 10;

    vector<int> iteration_thresholds = {
        detection_threshold, 
        major_pair_limit,
        accuracy
    };

    // Thresholds for minimum ellipse sizes
    double min_major = 50;
    double min_minor = 30;

    // Thresholds for merging ellipses
    double center_distance_threshold = 10;
    double semimajor_axis_threshold = 30;
    double semiminor_axis_threshold = 30;
    double angle_threshold = 50;


    vector<double> size_thresholds = {
        min_major, 
        min_minor,
        center_distance_threshold,
        semimajor_axis_threshold,
        semiminor_axis_threshold,
        angle_threshold
    };

    return make_tuple(iteration_thresholds, size_thresholds);
}

void drawEllipse(Mat &image, Mat thresholdedMag, 
    vector<Ellipse> hough_ellipse) {

    Mat image_clone = image.clone();
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
        ellipse(image_clone, center, axes, alpha, 0, 360,
            Scalar(0, 0, 255),
            2);
    }

    thresholdedMag.convertTo(thresholdedMag, CV_8U);

    Mat overlay;
    overlay.zeros(image_clone.size(), image_clone.type());
    ellipses.copyTo(overlay, thresholdedMag);

    imwrite("result/ellipses.jpg", ellipses);
    imwrite("result/overlay.jpg", overlay);
    imwrite("result/foundEllipses.jpg", image_clone);

}

void drawLines(Mat &image, Mat thresholdedMag, std::vector<double> &rhoValues, std::vector<double> &thetaValues) {

    Mat lines(image.size(), image.type(), Scalar(0));
    Mat image_clone = image.clone();

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
        line(image_clone, point1, point2, Scalar(0, 0, 255), 1);
    }

    thresholdedMag.convertTo(thresholdedMag, CV_8U);

    Mat overlay;
    overlay.zeros(image_clone.size(), image_clone.type());
    lines.copyTo(overlay, thresholdedMag);

    imwrite("result/lines.jpg", lines);
    imwrite("result/overlay.jpg", overlay);
    imwrite("result/foundLines.jpg", image_clone);
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

vector<Ellipse> houghEllipse(Mat &thresholdMag, int width, int height, tuple<vector<int>, vector<double>> ellipses_thresholds) {
    // Hough Ellipse detection based on 
    // Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection method." Pattern Recognition, 2002. Proceedings. 16th International Conference on. Vol. 2. IEEE, 2002

    srand( time( NULL ) );

    // Find all edges coordinates by using the thresholded magnitude
    vector<Point> locations;    
    Mat mag = thresholdMag.clone();
    mag.convertTo(mag, CV_8U);
    findNonZero(mag, locations);

    int detection_threshold = get<0>(ellipses_thresholds)[0];
    int major_pair_limit = get<0>(ellipses_thresholds)[1];
    int accuracy = get<0>(ellipses_thresholds)[2];

    double min_major = get<1>(ellipses_thresholds)[0];
    double min_minor = get<1>(ellipses_thresholds)[1];
    double center_distance_threshold = get<1>(ellipses_thresholds)[2];
    double semimajor_axis_threshold = get<1>(ellipses_thresholds)[3];
    double semiminor_axis_threshold = get<1>(ellipses_thresholds)[4];
    double angle_threshold = get<1>(ellipses_thresholds)[5];

    // Stores found ecllipse in (x0, y0, a, b, alpha, score) format
    vector<Ellipse> found_ellipse;

    // Iterate on all edge coordinates 
    for (int m = 0; m < locations.size() - 2; m++) {
        
        // Get first pixel to lookup
        int x1 = locations[m].x;
        int y1 = locations[m].y;

        for (int n = 0; n < major_pair_limit; n++) {

            // Get random choice of second pixel
            int randomIndex = m;

            while (randomIndex == m) {
                randomIndex = rand() % locations.size();          
            }

            // Get second pixel to lookup
            int x2 = locations[randomIndex].x;
            int y2 = locations[randomIndex].y;

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


            Mat accumulator(2,
                (int)((sqrt(pow(width,2) + pow(height,2)) + 1)/accuracy), 
                CV_64F, Scalar(0));

            for (int o = m + 1; o < locations.size(); o++) {

                if (o == randomIndex) continue;

                // Get third pixel to lookup
                int x = locations[o].x;
                int y = locations[o].y;

                double d = sqrt(pow(x - x0, 2) + pow(y - y0, 2));
                if (d < min_minor) continue;
                if (d > a) continue;

                // Get half-length of minor-axis (b)
                double f_square = pow(x - x1, 2) + pow(y - y1, 2);
                double cos_tau_square = pow(
                    (pow(a,2) + pow(d,2) - f_square)/(2*a*d)
                    ,2);

                // Assume b > 0 and avoid division by 0
                double k = pow(a,2) - pow(d,2) * cos_tau_square;
                if (k > 0 && cos_tau_square < 1) {
                    int b = cvRound(
                        sqrt((pow(a,2) * pow(d,2) * (1 - cos_tau_square))/k)/accuracy
                        );
                    if (b > min_minor/accuracy) accumulator.at<double>(0, b)++;
                }
                
            }

            double max;
            Point maxLoc;

            // Adds ellipse if the maximum of the accumulator exceeds threshold
            minMaxLoc(accumulator, NULL, &max, NULL, &maxLoc);


            Ellipse new_ellipse (Point(x0, y0), a, maxLoc.x * accuracy, alpha, max);


            Mat mask(mag.size(), CV_8U, Scalar(0));

            // Draw added ellipse on mask
            Size axes(a, maxLoc.x * accuracy);

            ellipse(mask, Point(x0, y0), axes, alpha, 0, 360,
                Scalar(255),
                2);

            Mat new_mag;
            mag.copyTo(new_mag, mask);

            int count = countNonZero(new_mag);

            
            if (count > detection_threshold) 
            {          
                // Thresholds for accumulator merging
                double center_distance_threshold = 10;
                double semimajor_axis_threshold = 30;
                double semiminor_axis_threshold = 30;
                double angle_threshold = 50;
                
                merge_ellipses(
                        found_ellipse, new_ellipse, 
                        center_distance_threshold, semimajor_axis_threshold,
                        semiminor_axis_threshold, angle_threshold
                    );

            }

        }
    }


    return found_ellipse;
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
