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
#include <fstream>
#include <stdio.h>
#include <tr1/unordered_map>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>

using namespace std::tr1;
using namespace std;
using namespace cv;

#define MERGE_THRESHOLD 0.15
#define IOU_THRESHOLD 0.3

/** Function Headers */
vector<Rect> detectAndDisplay( Mat frame, vector<Rect> ground_truths, bool merge_boxes);
vector<Rect> merge_boxes(const vector<Rect> faces);
float f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, string output, string img_name, float threshold);
vector<Rect> getGroundTruthsFromCSV(string csv);
string get_csv_file(const string fileExtension, const string filePrefix, const string imgName);

/** Global variables */
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
	char cwd[PATH_MAX];
	getcwd(cwd, sizeof(cwd));
	string dir(cwd);
	string input_folder = dir + "/images/";
	string cascade_name = dir + "/dartcascade/best.xml";
	string output_folder = dir +"/annotated/dartboard/";
	string ground_truth_folder = dir + "/CSVs/dartboard/";
	bool merge = true;
	string annotation_file_ext = "points.csv";

	int opt;

	while ((opt = getopt(argc, (char **) argv, "mi:c:o:a:")) != -1) {
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
            case 'm':
            	merge = false;
            	break;
	        default: /* '?' */
            	string usage = "\nUsage: " + string(argv[0]);
            	usage += " [-i input_folder] [-c cascade_xml] [-o output_folder] [-a annotations_folder] [-m]\n\n";
            	usage += "Options:\n";
            	usage += "-i input_folder, specifies image folder\n";
            	usage += "-c cascade_xml, specifies trained cascade.xml file to use\n";
            	usage += "-o output_folder, specifies output folder for results\n";
            	usage += "-a annotations_folder, specifies folder containing annotations in csv files\n";
            	usage += "-m, flag to set if output boxes of Viola-Jones to be merged\n";
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


    // Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    for (auto i: files) {

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
		vector<Rect> detected = detectAndDisplay( frame, ground_truths, merge );

		f1_test(detected, ground_truths, results_output, i, IOU_THRESHOLD);

		// 4. Save Result Image
		imwrite( output_folder + i , frame );
    	
    }
}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame, vector<Rect> ground_truths, bool merge )
{
	std::vector<Rect> boxes;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, boxes, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// Partitions bounding boxes
	if (merge) {
		boxes = merge_boxes(boxes);
	}


       // 4. Draw box around faces found
	for( int i = 0; i < boxes.size(); i++ )
	{
		rectangle(frame, Point(boxes[i].x, boxes[i].y), Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height), Scalar( 0, 255, 0 ), 2);
	}

	for (auto gt : ground_truths) 
	{
		//draw a red bounding box
        rectangle(frame, Point(gt.x, gt.y), Point(gt.x + gt.width, gt.y + gt.height),
                  Scalar(0, 0, 255), 2);
	}

	return boxes;


}

vector<Rect> merge_boxes(const vector<Rect> boxes) 
{
	// Partitions bounding boxes if IOU is above threshold
	unordered_map<int, set<int>*> partitions;

	// Iterate over each pair of boxes
	for( int i = 0; i < boxes.size(); i++ )
	{
		unordered_map<int,set<int>*>::iterator i_it = partitions.find(i);
		if (i_it == partitions.end()) 
		{
			// Insert default partitions
			set<int>* initial_set = new set<int>();
			initial_set->insert(i);
			i_it = partitions.insert(pair<int,set<int>*>(i, initial_set)).first;
		}

		for (int j = i + 1; j < boxes.size(); j++) 
		{
			// If IOU is above threshold, we partition them
			Rect intersection = boxes[i] & boxes[j];
			Rect box_union = boxes[i] | boxes[j];
			float intersectionArea = intersection.area();
			float unionArea = box_union.area();

			if ((intersectionArea / unionArea) > MERGE_THRESHOLD) 
			{	
				unordered_map<int,set<int>*>::iterator j_it = partitions.find(j);
				if (i_it != partitions.end() && j_it != partitions.end())
				{
					if (i_it->second != j_it->second) 
					{
						// Merge sets if pointers are of different partitions
						i_it->second->insert(j_it->second->begin(), j_it->second->end());					
						
						set<int> temp = *j_it->second;

						for (auto index : temp)
						{	
							// Change all pointers from partition of j to the new merged set
							partitions[index] = i_it->second;
						}

					}

				}
				else if (i_it != partitions.end())
				{
					// Add j to partition i if j is not partitioned
					i_it->second->insert(j);
					partitions[j] = i_it->second;

				}	
				else if (j_it != partitions.end())
				{
					// Add i to partition j if i is not partitioned
					j_it->second->insert(i);
					partitions[i] = j_it->second;
				}

			}

		}

	}

	vector<Rect> partitioned_boxes;

	for (auto elem : partitions) 
	{
		// Check if partition has been processed
		int n = elem.second->size();
		if (n > 0) {

			// Find average position, width and height of Rect in partition	
			Point pos = Point(0,0);
			int width = 0, height = 0;

			for (auto j : *elem.second) 
			{
				Rect box = boxes[j];
				pos += Point(box.x, box.y);
				width += box.width;
				height += box.height;
			}
			
			pos = Point(cvRound(pos.x/n), cvRound(pos.y/n));
			width = cvRound(width/n);
			height = cvRound(height/n);
			Rect avg_rect = Rect(pos, pos + Point(width, height));
			
			// Clears partition and push average rectangle
			partitioned_boxes.push_back(avg_rect);
			elem.second->clear();
		}	
	}

	return partitioned_boxes;
}

float f1_test(vector <Rect> &detected, vector <Rect> &ground_truth, string output, string img_name, float threshold) {
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
                    if (matched.find(j) == matched.end())
                    {
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

    float precision = (float) truePositives / ((float) truePositives + (float) falsePositives);
    float recall = (float) truePositives / (float) ground_truth.size();
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

    return f1;
}

vector<Rect> getGroundTruthsFromCSV(string csv) {
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
