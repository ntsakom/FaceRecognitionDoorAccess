/*
@ntsako maringa
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

//Function declarations
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
	// These vectors hold the images and corresponding labels
	vector<Mat> images;
	vector<int> labels;
	// Read in the data
	try {
		read_csv(csv, images, labels);		//read csv file, capture all images and their labels
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << csv << "\". Reason: " << e.msg << endl;
		exit(1);	//error opening training set, exit
	}
	
	// Train the facerecogniser
	model->train(images, labels);
	
	//Load classifier for face prediction
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

	// Matrix to hold video frame
	Mat frame;

	for (;;) {
		//capture current frame
		cap >> frame;
		// create clone of current frame
		Mat original = frame.clone();
		// Convert frame to grayscale
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Identify faces in frame and store them in vector
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);

		for (int i = 0; i < faces.size(); i++) {
			// Process face by face:
			Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);
			//resize face image to the size of the training set (necessary for eigen and fisher faces)
			Mat face_resized;
			cv::resize(face, face_resized, Size(50, 50), 1.0, 1.0, INTER_CUBIC);

			// Now perform the prediction
			int predictionLabel = -1;
			double confidence = 0.0;
			model->predict(face_resized, predictionLabel, confidence);
	
	//		rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			//Create the text we will annotate the box with:
			string box_text = format("Prediction = %d", predictionLabel);
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		}
		imshow("face_recognizer", original);

		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
}

