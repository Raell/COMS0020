#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <array>
#include <fstream>
#include <unordered_set>
#include <unistd.h>
#include <sys/types.h>

#define PI 3.14159265

#define GRADIENT_THRESHOLD 200

#define CANNY_LOW_THRESHOLD 100
#define CANNY_HIGH_THRESHOLD CANNY_LOW_THRESHOLD*3
#define ANGLE_RANGE 30
#define MIN_LINES_IN_DARTBOARD 5

#define MERGE_THRESHOLD 0.15
#define IOU_THRESHOLD 0.3

using namespace cv;
using namespace std;

class Line {
public:
    int rho;
    int theta;
    Point p1;
    Point p2;

    Line() : rho(0), theta(0) {}

    Line(int rho, int theta, Point p1, Point p2) {
        this->rho = rho;
        this->theta = theta;
        this->p1 = p1;
        this->p2 = p2;
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

bool compareByArea(const Circle &c1, const Circle &c2) {
    return c1.r < c2.r;
}

bool compareByX(const Circle &c1, const Circle &c2) {
    return c1.x < c2.x;
}

bool compareByY(const Circle &c1, const Circle &c2) {
    return c1.y < c2.y;
}

bool compareByTheta(const Line &l1, const Line &l2) {
    return l1.theta < l2.theta;
}

std::ostream &operator<<(std::ostream &strm, const Circle &circle) {
    return strm << "Circle: x= " << circle.x << " y=" << circle.y << " r=" << circle.r;
}

std::ostream &operator<<(std::ostream &strm, const Line &line) {
    return strm << "Line: rho= " << line.rho << " theta=" << line.theta;
}

vector <Circle> filter_circles(vector <Circle> &circles) {
    auto stringify = [](const pair<int, int> &p, string sep = "-") -> string {
        return to_string(p.first) + sep + to_string(p.second);
    };

    unordered_set <string> circles_seen;
    vector <Circle> distinct;
    for (const auto &c: circles) {
        auto centre = stringify(make_pair(c.x, c.y));
        if (circles_seen.find(centre) == circles_seen.end()) {
            distinct.push_back(c);
            circles_seen.insert(centre);
        }
    }
    return distinct;
}

string get_csv_file(string filePrefix, const string fileExtension, const string imgName);

vector <Line> filter_lines_by_theta(vector <Line> &lines, int threshold) {
    std::sort(lines.begin(), lines.end(), compareByTheta);
    auto window_begin = 0;
    auto window_end = 0;
    vector <Line> distinct;

    //maintains a sliding window to capture the range of lines 
    // satisfying the threshold
    while (window_end < lines.size()) {
        if (lines[window_begin].theta + threshold < lines[window_end].theta) {
            distinct.push_back(lines[window_begin]);
            window_begin = window_end;
        }
        window_end += 1;
    }

    return distinct;
}

CascadeClassifier cascade;

vector <Rect> getGroundTruthsFromCSV(string csv);
string get_csv_file(const char *imgName);
vector <Rect> detectAndDisplay(Mat frame, bool merge);

vector <Rect> merge_boxes(const vector <Rect> boxes);

tuple<float, float> f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, 
    string output, string img_name, float threshold);

Mat convolution(Mat &input, int direction, Mat kernel, cv::Size image_size);

void drawLines(Mat &image, Mat thresholdedMag, std::vector <Line> &detected_lines);
void drawCircles(Mat &image, vector <Circle> &circles);
void drawDetections(Mat &image, vector<Rect> detections, vector<Rect> ground_truths);

Rect contractBox(Rect &box);

Mat getMagnitude(Mat &dfdx, Mat &dfdy, cv::Size image_size);
Mat getDirection(Mat &dfdx, Mat &dfdy, cv::Size image_size);
void getThresholdedMag(Mat &input, Mat &output, double gradient_threshold);
Mat get_houghSpaceLines(Mat &thresholdMag, Mat &gradientDirection, int width, int height);

vector <Line> collect_lines_from_houghSpace(Mat &houghSpace, double threshold, Rect &box);
bool linesPassThroughBoxCentre(vector <Line> &lines, Rect &box, int threshold);

double calculate_houghLines_voting_threshold(Mat &hough_space);
double calculate_houghCircles_voting_threshold(std::vector < std::vector < std::vector < int >> > &hough_space);

double calculate_houghCircles_voting_threshold(std::vector < std::vector < std::vector < int >> > &hough_space) {
    return 14;
}

vector <Rect> pipeline(Mat &frame);

double calculate_houghLines_voting_threshold(Mat &hough_space) {
    double max, min;
    cv::minMaxLoc(hough_space, &min, &max);
    double houghSpaceThreshold = min + ((max - min) / 2) + 10;
    return houghSpaceThreshold;
}

bool concentricCircles(vector <Circle> &circles, const int threshold) {
    std::sort(circles.begin(), circles.end(), compareByX);
    auto max_x = circles.back().x;
    auto min_x = circles.front().x;
    if (max_x - min_x > threshold) {
        return false;
    }
    std::sort(circles.begin(), circles.end(), compareByY);
    auto max_y = circles.back().y;
    auto min_y = circles.front().y;
    if (max_y - min_y > threshold) {
        return false;
    }

    return true;
}

bool circlesAreContained(vector <Circle> &circles, int threshold = 20) {
    auto biggest_circle = std::max_element(circles.begin(), circles.end(), compareByArea);
    for (const auto &c: circles) {
        auto distance_bw_centres = (int) sqrt(pow(c.x - biggest_circle->x, 2) + pow(c.y - biggest_circle->y, 2));
        if (biggest_circle->r + threshold < distance_bw_centres + c.r) {
            return false;
        }

    }
    return true;

}

bool dartboardDetected(vector <Circle> &circles, vector <Line> &lines, Rect &box) {
    /*
     * A greedy heuristic algorithm that confirms presence of a dartboard, by exploiting results from the viola jones detection, the box merging algorithm, and general dartboard characteristics.
     * Given that the TPR of VJ is relatively high, we can make optimistic greedy decision choices.
     *
     * (1) we check if the region  has any circles at all. If it doesn't, then we ensure that it has at least 6 points passing through the rectangle centre
     *
     * (2) If it does have a high number of circles, check if the largest circle spans around half the area of the bounding box. If true, dartboard detected.
     *
     * (3) If the # of circles is not high enough, we check if all of those circles are contained in each other. If they are not, we have not detected a dartboard
     *
     * (4) As a last resort, check if there are a large number of lines passing through the centre of the contracted rectangle. If true, then a dartboard is detected.
     *
     * (5) If none of the conditions above are met, there is no dartboard
     *
     */

    circles = filter_circles(circles);
    lines = filter_lines_by_theta(lines, 25);

    cout << "circles detected: " << circles.size() << endl;
    cout << "lines detected: " << lines.size() << endl;

    for (auto &l: lines) {
        cout << l << endl;
    }

    if (linesPassThroughBoxCentre(lines, box, MIN_LINES_IN_DARTBOARD) && lines.size() >= MIN_LINES_IN_DARTBOARD) {
        return true;
    }

    if (circles.size() == 0) {
        return false;
    }

    auto biggest_circle = std::max_element(circles.begin(), circles.end(), compareByArea);
    auto max_circle_area = biggest_circle->area();
    auto box_area = box.area();

    if (circles.size() >= 50 && max_circle_area > box.area() / 2.5) {
        return true;
    }

    if (max_circle_area > box.area() / 2.5 && circlesAreContained(circles)) {
        return true;
    }

    return false;
}

bool linesPassThroughBoxCentre(vector <Line> &lines, Rect &box, int threshold) {
    auto reduced_box = contractBox(box);
    auto count = 0;

    int boxWidthBound = reduced_box.x + reduced_box.width;
    int boxHeightBound = reduced_box.y + reduced_box.height;

    for (const auto &line: lines) {
        double midpointX = (line.p1.x + line.p2.x) / 2;
        double midpointY = (line.p1.y + line.p2.y) / 2;

        bool xInBox = midpointX >= reduced_box.x && midpointX <= boxWidthBound;
        bool yInBox = midpointY >= reduced_box.y && midpointY <= boxHeightBound;

        if (xInBox && yInBox) {
            count++;
        }
    }

    return count >= threshold;


}

void drawCircles(Mat &image, vector <Circle> &circles) {
    for (const auto &c: circles) {
        Point center(c.x, c.y);
        cv::circle(image, center, c.r, Scalar(0, 0, 255), 2, 8, 0);
    }

    //  imwrite("result/foundCircles.jpg", image);
}

vector <Circle> houghCircles(Mat &image, Mat &thresholdMag, Mat &gradient_dir) {

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
    auto voting_threshold = calculate_houghCircles_voting_threshold(houghSpace);
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

vector <Rect> pipeline(Mat &frame, bool canny, bool merge) {

    vector <Rect> best_detections;

    Mat dxKernel = (Mat_<double>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);

    Mat dyKernel = (Mat_<double>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1);

    int counter = 0;



    //get viola-jones detections and draw then in GREEN.
    auto violaJonesDetections = detectAndDisplay(frame, merge);

    imwrite("result/violaJonesDetections.jpg", frame);

    //for every detection, apply hough transforms to find the number of circles and lines
    for (auto &rect: violaJonesDetections) {

        auto rgb_viola_jones = frame(rect);
        Mat gray_viola_jones;
        cvtColor(rgb_viola_jones, gray_viola_jones, CV_BGR2GRAY);

        Mat dfdx = convolution(gray_viola_jones, 0, dxKernel, gray_viola_jones.size());
        Mat dfdy = convolution(gray_viola_jones, 1, dyKernel, gray_viola_jones.size());

        Mat gradientDirection = getDirection(dfdx, dfdy, gray_viola_jones.size());
        imwrite("dir.jpg", gradientDirection);

        Mat thresholdedMag;
        thresholdedMag.create(gray_viola_jones.size(), CV_64F);

        if (canny)
        {
            Mat gradientMagnitude;
            blur(gray_viola_jones, gradientMagnitude, Size(3,3));
            Canny( gradientMagnitude, gradientMagnitude, 
                CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, 3);

            gradientMagnitude.convertTo(gradientMagnitude, CV_64F);

            thresholdedMag = gradientMagnitude;

        }
        else 
        {
            Mat gradientMagnitude = getMagnitude(dfdx, dfdy, gray_viola_jones.size());
            getThresholdedMag(gradientMagnitude, thresholdedMag, GRADIENT_THRESHOLD);
        }

        imwrite("magnew.jpg", thresholdedMag);

        auto circles = houghCircles(gray_viola_jones, thresholdedMag, gradientDirection);

        auto houghSpace = get_houghSpaceLines(thresholdedMag, gradientDirection, gray_viola_jones.cols,
                                              gray_viola_jones.rows);

        imwrite("houghnew.jpg", houghSpace);

        auto houghLinesThreshold = calculate_houghLines_voting_threshold(houghSpace);
        auto lines = collect_lines_from_houghSpace(houghSpace, houghLinesThreshold, rect);

        cout << "lines found: " << lines.size() << endl;
        cout << "circles found: " << circles.size() << endl;

        if (dartboardDetected(circles, lines, rect)) {
            counter += 1;
            cout << "result: detected" << endl;

            drawLines(rgb_viola_jones, thresholdedMag, lines);
            drawCircles(rgb_viola_jones, circles);
            best_detections.push_back(rect);

        } else {
            cout << "result: NOT detected" << endl;
            drawLines(rgb_viola_jones, thresholdedMag, lines);
        }
        cout << "######################" << std::endl;

    }

    cout << "Total dartboards detected: " << counter << endl;

    imwrite("result/detections.jpg", frame);
    return best_detections;
}

int main(int argc, const char **argv) {

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    string dir(cwd);
    string input_folder = dir + "/images/";
    string cascade_name = dir + "/dartcascade/best.xml";
    string output_folder = dir +"/annotated/dartboard/";
    string ground_truth_folder = dir + "/CSVs/dartboard/";
    bool merge = true;
    bool canny = true;
    bool use_pipeline = true;
    string annotation_file_ext = "points.csv";

    int opt;

    while ((opt = getopt(argc, (char **) argv, "p:i:c:o:a:")) != -1) {
        switch (opt) {
            case 'i':
                input_folder = optarg;
                break;
            case 'c':
                cascade_name = optarg;
                break;
            case 'o':
                output_folder = optarg;
                break;
            case 'a':
                ground_truth_folder = optarg;
                break;
            case 'p': {
                if (strcmp(optarg, "FULL") == 0) {
                    break;
                }
                else if (strcmp(optarg, "CANNY") == 0) {
                    merge = false;
                    break;
                }
                else if (strcmp(optarg, "MERGE") == 0) {
                    canny = false;
                    break;
                }
                else if (strcmp(optarg, "BASIC") == 0) {
                    merge = false;
                    canny = false;
                    break;
                }
                else if (strcmp(optarg, "VJ") == 0) {
                    merge = false;
                    canny = false;
                    use_pipeline = false;
                    break;
                }
            }
            default: /* '?' */
                string usage = "\nUsage: " + string(argv[0]);
                usage += " [-i input_folder] [-c cascade_xml] [-o output_folder] [-a annotations_folder] [-p PIPELINE]\n\n";
                usage += "Options:\n";
                usage += "-i input_folder, specifies image folder\n";
                usage += "-c cascade_xml, specifies trained cascade.xml file to use\n";
                usage += "-o output_folder, specifies output folder for results\n";
                usage += "-a annotations_folder, specifies folder containing annotations in csv files\n";
                usage += "-p PIPELINE, flag to set what pipeline is run\n";
                usage += "\tParameters:\n";
                usage += "\tFULL - entire detection pipeline with merge and canny edges (default)\n";
                usage += "\tCANNY - detection pipeline with canny edges\n";
                usage += "\tMERGE - detection pipeline with merge\n";
                usage += "\tBASIC - basic detection pipeline\n";
                usage += "\tVJ - only use Viola-Jones detector\n";
                cerr << usage << endl;
                exit(EXIT_FAILURE);
        }
    }

    // Read the input folder
    Vector<string> files;
    for (int i = 0; i <= 15; i++) {
        files.push_back("dart" + to_string(i) + ".jpg");
    }

    string results_output = output_folder + "results.txt";
    // Clear file if exists
    remove(results_output.c_str());

    float avg_tpr = 0;
    float avg_f1 = 0;

    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    for (auto i: files)
    {     
        // 1. Read Input Image
        Mat frame = imread(input_folder + i, CV_LOAD_IMAGE_COLOR);

        if(!frame.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image " << input_folder + i << std::endl ;
            return -1;
        }

        string csv_file_path = get_csv_file(ground_truth_folder, annotation_file_ext, i);

        vector<Rect> ground_truths = getGroundTruthsFromCSV(csv_file_path);

        // 3. Detect Faces and Display Result
        vector<Rect> best_detections;
    

        Mat frame_clone = frame.clone();

        if (use_pipeline) {
            best_detections = pipeline(frame_clone, canny, merge);
        }
        else {
            best_detections = detectAndDisplay(frame_clone, merge);
        }
        // vector<Rect> detected = detectAndDisplay( frame, ground_truths, merge );

        tuple<float,float> results = f1_test(best_detections, ground_truths, results_output, i, IOU_THRESHOLD);

        avg_tpr += get<0>(results);
        avg_f1 += get<1>(results);

        // 4. Save Result Image
        drawDetections(frame, best_detections, ground_truths);
        imwrite( output_folder + i , frame );
        
    }

    avg_tpr /= 16;
    avg_f1 /= 16;

    ofstream output_file;
    output_file.open(results_output, fstream::app);
    output_file << "Avg TPR: " << avg_tpr << "\n";
    output_file << "Avg F1: " << avg_f1 << "\n\n";
    output_file.close();
}

void drawDetections(Mat &image, vector<Rect> detections, vector<Rect> ground_truths) {
    for(auto det : detections)
    {
        rectangle(image, Point(det.x, det.y), 
            Point(det.x + det.width, det.y + det.height), 
            Scalar( 0, 255, 0 ), 2);
    }

    for (auto gt : ground_truths) 
    {
        //draw a red bounding box
        rectangle(image, Point(gt.x, gt.y), 
            Point(gt.x + gt.width, gt.y + gt.height),
                  Scalar(0, 0, 255), 2);
    }
}

void drawLines(Mat &image, Mat thresholdedMag, std::vector <Line> &detected_lines) 
{

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
                                            double threshold, Rect &box) {
    /*
     * Populates the line vector & thresholds the houghspace.
     */
    std::vector <Line> lines;

    for (int rho = 0; rho < houghSpace.rows; rho++) {
        for (int theta = 0; theta < houghSpace.cols; theta++) {
            double val = houghSpace.at<double>(rho, theta);

            if (val > threshold) {
                Point point1, point2;

                double radians = theta * (PI / 180);
                double m = cos(radians) / sin(radians);
                double c = (rho - box.width - box.height) / sin(radians);

                point1.x = cvRound(box.x);
                point1.y = cvRound(box.y) + cvRound(c);
                // When x = end of image
                point2.x = cvRound(box.x + box.width);
                point2.y = cvRound(box.y) + cvRound(c - (box.width * m));

                clipLine(box, point1, point2);
                auto l = Line(rho, theta, point1, point2);

                lines.push_back(l);
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
            if (value >= 255) {
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

Rect contractBox(Rect &box) {

    double originalArea = box.width * box.height;
    double halfArea = originalArea / 4;
    double newLength = cvRound(sqrt(halfArea));

    double centreX = box.x + (box.width / 2);
    double centreY = box.y + (box.height / 2);

    double newCentreX = cvRound(centreX - (newLength / 2));
    double newCentreY = cvRound(centreY - (newLength / 2));

    Rect newBox(newCentreX, newCentreY, newLength, newLength);

    return newBox;
}

vector <Rect> detectAndDisplay(Mat frame, bool merge) {
    Mat frame_gray;
    vector <Rect> detected;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);


    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, detected, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    cout << "boxes found: " << detected.size() << std::endl;

    vector<Rect> merged = detected;

    // 2.5 Merge overlapping rectangles
    if (merge) {
        merged = merge_boxes(detected);
        cout << "boxes after merging: " << merged.size() << std::endl;
    }
    

    return merged;
}

vector <Rect> merge_boxes(const vector <Rect> boxes) {
    // Partitions bounding boxes if IOU is above threshold
    unordered_map < int, set < int > * > partitions;

    // Iterate over each pair of boxes
    for (int i = 0; i < boxes.size(); i++) {
        unordered_map < int, set < int > * > ::iterator
        i_it = partitions.find(i);
        if (i_it == partitions.end()) {
            // Insert default partitions
            set<int> *initial_set = new set<int>();
            initial_set->insert(i);
            i_it = partitions.insert(pair < int, set < int > * > (i, initial_set)).first;
        }

        for (int j = i + 1; j < boxes.size(); j++) {
            // If IOU is above threshold, we partition them
            Rect intersection = boxes[i] & boxes[j];
            Rect box_union = boxes[i] | boxes[j];
            float intersectionArea = intersection.area();
            float unionArea = box_union.area();

            if ((intersectionArea / unionArea) > MERGE_THRESHOLD) {
                unordered_map < int, set < int > * > ::iterator
                j_it = partitions.find(j);
                if (i_it != partitions.end() && j_it != partitions.end()) {
                    if (i_it->second != j_it->second) {
                        // Merge sets if pointers are of different partitions
                        i_it->second->insert(j_it->second->begin(), j_it->second->end());

                        set<int> temp = *j_it->second;

                        for (auto index : temp) {
                            // Change all pointers from partition of j to the new merged set
                            partitions[index] = i_it->second;
                        }

                    }

                } else if (i_it != partitions.end()) {
                    // Add j to partition i if j is not partitioned
                    i_it->second->insert(j);
                    partitions[j] = i_it->second;

                } else if (j_it != partitions.end()) {
                    // Add i to partition j if i is not partitioned
                    j_it->second->insert(i);
                    partitions[i] = j_it->second;
                }

            }

        }

    }

    vector <Rect> partitioned_boxes;

    for (auto elem : partitions) {
        // Check if partition has been processed
        int n = elem.second->size();
        if (n > 0) {

            // Find average position, width and height of Rect in partition
            Point pos = Point(0, 0);
            int width = 0, height = 0;

            for (auto j : *elem.second) {
                Rect face = boxes[j];
                pos += Point(face.x, face.y);
                width += face.width;
                height += face.height;
            }

            pos = Point(cvRound(pos.x / n), cvRound(pos.y / n));
            width = cvRound(width / n);
            height = cvRound(height / n);
            Rect avg_rect = Rect(pos, pos + Point(width, height));

            // Clears partition and push average rectangle
            partitioned_boxes.push_back(avg_rect);
            elem.second->clear();
        }
    }

    return partitioned_boxes;
}

tuple<float,float> f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, string output, string img_name, float threshold) {
    int truePositives = 0;
    int falsePositives = 0;
    set<int> matched;

    for (int i = 0; i < detected.size(); i++) {
        bool matchFound = false;

        for (int j = 0; j < ground_truth.size(); j++) {

            Rect intersection = detected[i] & ground_truth[j];
            Rect box_union = detected[i] | ground_truth[j];
            float intersectionArea = intersection.area();
            float unionArea = box_union.area();

            if (intersectionArea > 0) {
                float matchPercentage = (intersectionArea / unionArea);
                if (matchPercentage > threshold) {
                    if (matched.find(j) == matched.end()) {
                        truePositives++;
                        matched.insert(j);
                    }
                    matchFound = true;
                }
            }
        }
        if (!matchFound) {
            falsePositives++;
        }
    }

    // Precision = TP / (TP + FP)
    // Recall = TPR (True Positive Rate)
    // F1 = 2((PRE * REC)/(PRE + REC))
    int falseNegatives = ground_truth.size() - truePositives;

    float precision = (float) truePositives / (truePositives + falsePositives);
    float recall = (float) truePositives / ground_truth.size();
    float f1;
    if (precision > 0 && recall > 0) {
        f1 = 2 * (precision * recall) / (precision + recall);
    }
    else if (!truePositives && !falsePositives && !falseNegatives) {
        f1 = 1;
        recall = 1;
    }
    else {
        f1 = 0;
        recall = 0;
    }
    
    // Appends results to output file
    ofstream output_file;
    output_file.open(output, fstream::app);
    output_file << img_name << "\n";
    output_file << "total detected: " << detected.size() << ", ground truth: " << ground_truth.size() << "\n";
    output_file << "true positives: " << truePositives << ", false positives: " << falsePositives << "\n";
    output_file << "TPR: " << recall << "\n";
    output_file << "f1 score: " << f1 << "\n\n";
    output_file.close();

    return make_tuple(recall, f1);
}

vector <Rect> getGroundTruthsFromCSV(string csv) {
    const char *c = csv.c_str();
    vector <Rect> ground_truths;
    string current_line;
    ifstream inputFile(c);
    if (!inputFile.good()) {
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

string get_csv_file(const string filePrefix, const string fileExtension, const string imgName) {
    string current_line;
    string csv_filename(imgName);
    string::size_type i = csv_filename.rfind('.', csv_filename.length());
    if (i != string::npos) {
        csv_filename.replace(i, fileExtension.length(), fileExtension);
    }

    return filePrefix + csv_filename;
}
