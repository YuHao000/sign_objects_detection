#include "stdafx.h"
#include "mouse_click.h"
#include "detection_objects.h"
#include "auto_correct.h"
#include <ctime>
#include <map>

#include <cv.h>
#include <cxcore.h>

#ifndef _EiC
#include "cv.h"
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

#include <iostream>

using namespace std;

#define SEARCH_R 100
#define SEARCH_G 0
#define SEARCH_B 0
#define SEARCH_X 0
#define	SEARCH_Y 0
#define	CHECK 0
#define SEARCH_EPS 30
#define SEARCH_EPS1 40
#define SEARCH_EPS2 50
#define SEARCH_EPS3 60
#define ORIGINAL_IMAGE "Original image"
#define CORRECTION_IMAGE "Image after correction"
#define EQUALIZE_IMAGE "Image after equalize histogram"
#define IMAGE_AFTER_COLOR "Image after color handler"

//objects
#define OBJECT_ONE        0
#define OBJECT_ONE_NAME   "Main road sign"


int main(int argc, char* argv[])
{
    // Map with recognizable road signs
	std::map< short, std::string > objects_map;
    objects_map.emplace(MAIN_ROAD_SIGN, "Main road sign");
    objects_map.emplace(STOP_SIGN, "Stop sign");
    objects_map.emplace(GIVE_WAY_SIGN, "Give way sign");
    objects_map.emplace(CROSSWALK_SIGN, "Crosswalk sign");
    objects_map.emplace(OVERTAKING_SIGN, "Overtaking prohibited sign");
    objects_map.emplace(NO_PARK_SIGN, "No parking sign");
    objects_map.emplace(LEFT_SIGN, "Left sign");
    objects_map.emplace(RIGHT_SIGN, "Right sign");
    objects_map.emplace(FORWARD_SIGN, "Forward left sign");
    objects_map.emplace(FORWARD_LEFT_SIGN, "Forward right sign");
    objects_map.emplace(FORWARD_RIGHT_SIGN, "Left sign");
    objects_map.emplace(LEFT_RIGHT_SIGN, "Left right sign");
    objects_map.emplace(GARBAGE, "Garbage");
	std::string object_name = OBJECT_ONE_NAME;
	short object_idx = OBJECT_ONE;

	IplImage* frame = 0;
	IplImage* original = 0;
	IplImage* after_correct = 0;
	IplImage* equal_hist = 0;
	IplImage* color_image = 0;

	CvCapture* capture = cvCreateCameraCapture(0);
	assert(capture);

	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 960);

	cvNamedWindow(ORIGINAL_IMAGE, CV_WINDOW_NORMAL);
	cvNamedWindow(CORRECTION_IMAGE, CV_WINDOW_NORMAL);
	cvNamedWindow(EQUALIZE_IMAGE, CV_WINDOW_NORMAL);
	cvNamedWindow(IMAGE_AFTER_COLOR, CV_WINDOW_NORMAL);

    int width = static_cast<int>(cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT));

	MouseClick my_mouse(false, width, height);
	int epsilon = SEARCH_EPS;
	short mode = 0;
	int count_image = 0;
	bool show_garbage = true;

	ObjectsDetection objects_detection(epsilon, width, height);

	while (true)
    {
        frame = cvQueryFrame(capture);
		cvReleaseImage(&original);
		cvReleaseImage(&color_image);
		cvReleaseImage(&after_correct);
		cvReleaseImage(&equal_hist);
		original = cvCloneImage(frame);

		after_correct = AutoCorrect(original).GetResult();
		color_image = cvCloneImage(after_correct);

		objects_detection.SetImage(color_image);
		my_mouse.SetImage(after_correct);
		void* p_mouse = (void*)(&my_mouse);
		unsigned int start_time = 0;
		unsigned int end_time = 0;
		std::vector<int> time_list;

		switch (mode)
		{
		case 0:
            // Mode for collect color features
			cvSetMouseCallback(CORRECTION_IMAGE, MouseClick::MyMouseClick, p_mouse);
			objects_detection.AddOptionsToObject(my_mouse, object_idx, object_name);
			break;
		case 1:
		case 2:
            // Mode for color processing each image of video flow.
            // Also in this mode user can to select objects of interest.
            // Based on this objects, vector of textural features are formed.
			start_time = clock();
			objects_detection.ColorDetectedMass();
			end_time = clock();
			time_list.push_back(end_time - start_time);
			objects_detection.ShowContours(color_image);
			objects_detection.ShowContours(original);
			my_mouse.SetImage(color_image);
			my_mouse.SetCrackImage(original, objects_map[object_idx]);
			cvSetMouseCallback(IMAGE_AFTER_COLOR, MouseClick::MyMouseClickForTrain, p_mouse);
			objects_detection.AddTextureToObject(my_mouse, object_idx, object_name, mode);
			break;
		case 3:
            // Mode for train neural network.
			objects_detection.TrainNet();
			mode = 4;
			break;
		case 4:
            // Work mode of detector by color and textural features.
			objects_detection.Detected(original, show_garbage);
			break;
		}

		cvShowImage(ORIGINAL_IMAGE, original);
		cvShowImage(CORRECTION_IMAGE, after_correct);
		cvShowImage(IMAGE_AFTER_COLOR, color_image);
		char c = cvWaitKey(33);
		if (c == 27)
		{
			int sum = 0;
			std::for_each(time_list.begin(), time_list.end(), [&](int value) {
				sum += value;
			});
			std::cout << "ColorDetected time: " << sum / time_list.size() << "\n";
			break;
		}

		if (c == 'm')
		{
			object_idx = OBJECT_ONE;
			mode = 0;
			cout << "Main training mode" << "\n";
		}
		if (c == 't')
		{
			object_idx = OBJECT_ONE;
			mode = 1;
			cout << "Training mode" << "\n";
		}
		if (c == 'l')
		{
			mode = 2;
			cout << "Learning mode" << "\n";
		}
		if (c == 'w')
		{
			mode = 3;
			cout << "Working mode" << "\n";
		}

		if (c == 'n')
		{
			object_idx++;
			object_name = objects_map[object_idx];
			cout << "Object: " << object_name << "\n";
		}
		if (c == 'g') show_garbage = !show_garbage;
		if (c == 'c')
		{
			mode = 0;
			object_idx = 0;
			objects_detection.ClearOptions();
		}
		if (c == 'o')
		{
			mode = 0;
			object_idx = 0;
			objects_detection.ClearTextures();
		}
	}

	cvReleaseImage(&frame);
	cvReleaseImage(&original);
	cvReleaseImage(&after_correct);
	cvReleaseImage(&equal_hist);
	cvReleaseImage(&color_image);
	cvDestroyAllWindows();
	return 0;
}

