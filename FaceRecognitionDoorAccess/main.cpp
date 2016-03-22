/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

void infinite_loop();
void FaceRecognition();
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
//Global variables
CascadeClassifier haar_cascade; //classifier for the task of Face Detection
Ptr<FaceRecognizer> model = createFisherFaceRecognizer(0,50); //Face recognition class to perform recognition

int main(int argc, const char *argv[]) {
	//Setup
	// Get the path to CSV:
	string csv = "C:/Users/LenovoS510p/Documents/Semester1_2016/Computer_Engineering_Design3/Face_Recognition/fn_csv.csv";
	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;
	// Read in the data
	try {
		read_csv(csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	//int im_width = images[0].cols;
	//int im_height = images[0].rows;

	// Create a FaceRecognizer and train it on the given images:
	model->train(images, labels);
	
	//Load classifier
	haar_cascade.load("C:/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");

	//Goto infinite loop and wait for user input
	infinite_loop();
	return 0;
}

void infinite_loop(){
	for (;;) {
			FaceRecognition();
	}
}

void FaceRecognition() {
	// Get a handle to the Video device:
	VideoCapture cap(0);
	// Check if we can use this device at all:
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << "cannot be opened." << endl;
		exit(1);
	}

	// Holds the current frame from the Video/image device:
	Mat frame;

	for (;;) {
		cap >> frame;
		// Clone the current frame:
		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);

		for (int i = 0; i < faces.size(); i++) {
			// Process face by face:
			Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);
			// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
			// verify this, by reading through the face recognition tutorial coming with OpenCV.
			// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
			// input data really depends on the algorithm used.
			//
			// I strongly encourage you to play around with the algorithms. See which work best
			// in your scenario, LBPH should always be a contender for robust face recognition.
			//
			// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
			// face you have just found:
			Mat face_resized;
			cv::resize(face, face_resized, Size(50, 50), 1.0, 1.0, INTER_CUBIC);

			// Now perform the prediction
			int predictionLabel = -1;
			double confidence = 0.0;
			model->predict(face_resized, predictionLabel, confidence);
			//// And finally write all we've found out to the original image!
			//// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			//// Create the text we will annotate the box with:
			string box_text = format("Prediction = %d", predictionLabel);
			//// Calculate the position for annotated text (make sure we don't
			//// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			//// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		}
		imshow("face_recognizer", original);

		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
}

