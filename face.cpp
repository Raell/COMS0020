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
#include <tr1/unordered_map>

using namespace std::tr1;
using namespace std;
using namespace cv;

#define MERGE_THRESHOLD 0.15

/** Function Headers */
void detectAndDisplay( Mat frame );
vector<Rect> merge_faces(const vector<Rect> faces);

/** Global variables */
String cascade_name = "dartcascade/best.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	std::string filename(argv[1]);

    string::size_type i = filename.rfind('.', filename.length());
    if (i != string::npos) {
        filename.replace(i, 4, "");
    }


	// 4. Save Result Image
	imwrite( filename + "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// Partitions bounding boxes
	vector<Rect> partitioned_faces = merge_faces(faces);

	// Print out number of paritioned faces
	cout << partitioned_faces.size() << endl;


       // 4. Draw box around faces found
	for( int i = 0; i < partitioned_faces.size(); i++ )
	{
		rectangle(frame, Point(partitioned_faces[i].x, partitioned_faces[i].y), Point(partitioned_faces[i].x + partitioned_faces[i].width, partitioned_faces[i].y + partitioned_faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

vector<Rect> merge_faces(const vector<Rect> faces) 
{
	// Partitions bounding boxes if IOU is above threshold
	unordered_map<int, set<int>*> partitions;

	// Iterate over each pair of faces
	for( int i = 0; i < faces.size(); i++ )
	{
		unordered_map<int,set<int>*>::iterator i_it = partitions.find(i);
		if (i_it == partitions.end()) 
		{
			// Insert default partitions
			set<int>* initial_set = new set<int>();
			initial_set->insert(i);
			i_it = partitions.insert(pair<int,set<int>*>(i, initial_set)).first;
		}

		for (int j = i + 1; j < faces.size(); j++) 
		{
			// If IOU is above threshold, we partition them
			Rect intersection = faces[i] & faces[j];
			Rect box_union = faces[i] | faces[j];
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

	vector<Rect> partitioned_faces;

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
				Rect face = faces[j];
				pos += Point(face.x, face.y);
				width += face.width;
				height += face.height;
			}
			
			pos = Point(cvRound(pos.x/n), cvRound(pos.y/n));
			width = cvRound(width/n);
			height = cvRound(height/n);
			Rect avg_rect = Rect(pos, pos + Point(width, height));
			
			// Clears partition and push average rectangle
			partitioned_faces.push_back(avg_rect);
			elem.second->clear();
		}	
	}

	return partitioned_faces;
}
