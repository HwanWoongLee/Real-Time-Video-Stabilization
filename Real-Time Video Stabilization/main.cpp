#include <iostream>
#include <opencv2/opencv.hpp>
#include "stabilizer.h"


#pragma comment(lib, "opencv_world480.lib")

using namespace cv;
using namespace std;



int main() 
{
	// Input Video Path
	string strVideoPath = "";

	cout << "input video path : ";
	getline(cin, strVideoPath);

	if (strVideoPath.empty())
		return -1;

	// Craete stabilizer
	stabilizer stab;

	// Read Video File
	VideoCapture cap(strVideoPath);
	if (!cap.isOpened())
		return -1;

	Mat frame;
	Mat stab_frame, graph;
	Mat result_frame;

	while (cap.read(frame)) {
		if (frame.empty())
			break;

		// Stabilization
		stab_frame = stab.Stab(frame);
		graph = stab.GetGraph();
		
		hconcat(frame, stab_frame, result_frame);

		cv::imshow("result", result_frame);
		cv::imshow("graph", graph);

		cv::waitKey(10);
	}

	return 0;
}