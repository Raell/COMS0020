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
void houghLines(Mat mag, Mat dir, Mat &hough_sp);
void threshold(Mat &m, double t);
void normalize(Mat m);
void display(Mat m);
void overlay_shape(Mat mag, Mat hough_sp, Mat orig, Mat&overlay);

int main(int argc, char** argv)
{
    // Read Input Image
	char* mag_file = argv[1];
 	char* dir_file = argv[2];
 	char* orig_file = argv[3];

 	double mag_thresh = 60;

 	Mat mag = imread(mag_file, CV_LOAD_IMAGE_GRAYSCALE);
 	mag.convertTo(mag, CV_64F);
 	Mat dir = imread(dir_file, CV_LOAD_IMAGE_GRAYSCALE);
 	dir.convertTo(dir, CV_64F);
 	dir = dir * (2*CV_PI/255) - CV_PI;
 	Mat orig = imread(orig_file);
 	// orig.convertTo(orig, CV_64F);

	threshold(mag, mag_thresh);

 	Mat hough_sp;
	
	houghLines(mag, dir, hough_sp);

	// display(hough_sp);

	double min; 
	double max; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc(dir, &min, &max, &minLoc, &maxLoc);

	cout << "Min: " << min << endl;
	cout << "Max: " << max << endl;
	cout << "MinLoc: " << minLoc << endl;
	cout << "MaxLoc: " << maxLoc << endl;

	double hough_thresh = 190;

	threshold(hough_sp, hough_thresh);
	// display(hough_sp);

	Mat overlay;
	overlay_shape(mag, hough_sp, orig, overlay);
	namedWindow("Display", WINDOW_AUTOSIZE);
    imshow("Display", overlay);
    waitKey(0);  



	return 0;
}

void houghLines(Mat mag, Mat dir, Mat &hough_sp) {
	int p_space = (int)(sqrt(pow(mag.rows, 2) + pow(mag.cols, 2)) + 1);
	// int angle_range = 50;

	hough_sp = Mat::zeros(2 * p_space, 180, CV_64F);
	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0; x < mag.cols; x++) {
			if(mag.at<double>(y,x)) {
				// double grad = dir.at<double>(y,x);
				// double range = angle_range * (CV_PI/180);
				// int lower = cvRound((grad - range)*(180/CV_PI));
				// int upper = cvRound((grad + range)*(180/CV_PI));
				for(int theta = -90; theta < 90; theta++) {
					double rad = (double)theta/180 * CV_PI;
					int p = (int)((x * cos(rad) + y * sin(rad)));
					hough_sp.at<double>(p + p_space, theta+90) += 1;
				}
			}
		}
	}

}

void threshold(Mat &m, double t)
{
	for (int i = 0; i < m.rows; i++)
	{	
		for(int j = 0; j < m.cols; j++)
		{
			if (m.at<double>(i,j) < t) {
				m.at<double>(i,j) = (double) 0;
			}
			else {
				m.at<double>(i,j) = (double) 255;
			}
		}
	}
}

void normalize(Mat m) {
	double min; 
	double max; 
	minMaxLoc(m, &min, &max);

	m = (m - min) * (255.0/(max - min));
}


void display(Mat m) {
	Mat clone = m.clone();
	Mat f;
	flip(clone, f, 0);
	normalize(f);

	namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", f/255);
    waitKey(0);  
}

void overlay_shape(Mat mag, Mat hough_sp, Mat orig, Mat&overlay) {
	int p_space = (int)(sqrt(pow(orig.rows, 2) + pow(orig.cols, 2)) + 1);
	// overlay = Mat::zeros(orig.size(), CV_64F);
	overlay = orig.clone();
	for (int p = 0; p < hough_sp.rows; p++)
	{	
		for(int theta = 0; theta < hough_sp.cols; theta++)
		{
			if(hough_sp.at<double>(p,theta)) {
				int rho = p - p_space;
				double rad = (double)(theta-90)/180 * CV_PI;

				double a = cos(rad);
			    double b = sin(rad);
			    double x0 = a*rho;
			    double y0 = b*rho;
			    double x1 = cvRound(x0 + 1000*(-b));
			    double y1 = cvRound(y0 + 1000*(a));
			    double x2 = cvRound(x0 - 1000*(-b));
			    double y2 = cvRound(y0 - 1000*(a));

			    line(overlay, 
					Point(x1, y1), 
					Point(x2, y2),
					Scalar(0, 0, 255),
					1);
			}

		}
	} 
}