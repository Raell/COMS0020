// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void convolution(Mat image, Mat kernel, Mat &output);
void magnitude(Mat dx, Mat dy, Mat &mag);
void normalize(Mat m);
void direction(Mat dx, Mat dy, Mat &grad);

int main(int argc, const char** argv)
{
    // Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	frame.convertTo(frame, CV_64F);
	Mat x_kernel = (Mat_<double> (3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat y_kernel = (Mat_<double> (3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat dy;
	Mat dx;
	convolution(frame, x_kernel, dx);
	convolution(frame, y_kernel, dy);

	Mat mag;
	magnitude(dx, dy, mag);
	

	Mat dir;
	direction(dx, dy, dir);
	dir = (dir + CV_PI)/(2*CV_PI) * 255;

	double min; 
	double max; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc(dir, &min, &max, &minLoc, &maxLoc);

	cout << "Min: " << min << endl;
	cout << "Max: " << max << endl;
	cout << "MinLoc: " << minLoc << endl;
	cout << "MaxLoc: " << maxLoc << endl;

	normalize(dx);
	normalize(dy);
	normalize(mag);

	imwrite("sobelx.jpg", dx);
	imwrite("sobely.jpg", dy);
	imwrite("sobelmag.jpg", mag);
	imwrite("sobeldir.jpg", dir);

	return 0;
}

void convolution(Mat image, Mat kernel, Mat &output) {
	output.create(image.size(), image.type());
	for (int i = 0; i < image.rows - 2; i++) {
		for (int j = 0; j < image.cols - 2; j++) {
			output.at<double>(i + 1, j + 1) = sum(image(Rect(j, i, 3, 3)).mul(kernel))[0];
		}
	}
}

void magnitude(Mat dx, Mat dy, Mat &mag) {
	mag.create(dx.size(), CV_64F);

	sqrt(dx.mul(dx) + dy.mul(dy), mag);
}

void normalize(Mat m) {
	double min; 
	double max; 
	minMaxLoc(m, &min, &max);

	m = (m - min) * (255/(max - min));
}

void direction(Mat dx, Mat dy, Mat &dir) {
	dir.create(dx.size(), CV_64F);
	for (int i = 0; i < dir.rows; i++) {
		for (int j = 0; j < dir.cols; j++) {
			dir.at<double>(i, j) = atan2(dx.at<double>(i, j), dy.at<double>(i, j));

		}

	}

}