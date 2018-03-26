#pragma once

#include <cv.h>
#include <cxcore.h>
#include "dataanalysis.h"
#include <map>

#ifndef _EiC
#include "cv.h"
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

#define THRESHOLD 5000

enum RoadSignsIdx
{
	MAIN_ROAD_SIGN,
	STOP_SIGN,
	GIVE_WAY_SIGN,
	CROSSWALK_SIGN,
	OVERTAKING_SIGN,
	NO_PARK_SIGN,
	LEFT_SIGN,
	RIGHT_SIGN,
	FORWARD_SIGN,
	FORWARD_LEFT_SIGN,
	FORWARD_RIGHT_SIGN,
	LEFT_RIGHT_SIGN,
	GARBAGE
};

enum TrainIdx
{
	FIRST_MOMENT,
	SECOND_MOMENT,
	THIRD_MOMENT,
	FORTH_MOMENT,
	FIFTH_MOMENT,
	SIXTH_MOMENT,
	//SEVENTH_MOMENT,
	LAST_MOMENT
};

class MouseClick;
class Point;
class BaseObject
{
public:
	BaseObject(const MouseClick& my_mouse);
	int mRed;
	int mGreen;
	int mBlue;
	int mX;
	int mY;
};


class RoadSigns
{
public:
	RoadSigns(const std::string& object_name, int width, int height);
	std::vector< BaseObject* > mOptions;
	std::vector< Point* > mTexturesBase;
	std::vector< Point* > mTexturesTrain;
	std::string Name() { return mObjectName; }
	bool PointInEpsilon(uchar* ptr, int& epsilon, int& x);
	void Inc();
	void Clear();
private:
	std::string mObjectName;
	int mWidth;
	int mHeight;
	int mCount;
};


class ObjectsDetection
{
public:
	ObjectsDetection(int epsilon, int width, int height);
	void TrainNet();
	void SetImage(IplImage* image);
	void ColorDetectedMass();
	static bool ColorDetected(RoadSigns* road_sign, int height, int width, int eps, IplImage* img);
	void Detected(IplImage* original, bool show_garbage);
	void AddOptionsToObject(MouseClick& my_mouse, short idx, std::string& object_name);
	void AddTextureToObject(MouseClick& my_mouse, short idx,
		std::string& object_name, const short& mode);
	void ShowContours(IplImage* original);
	void ClearOptions();
	void ClearTextures();
private:
	std::map< short, RoadSigns* > mSignsList;
	int mWidth;
	int mHeight;
	IplImage* mColorImage;
	int mEpsilon;

	//для обучения нейронной сети
	alglib::multilayerperceptron mNet;
	int mNumberOfFeatures;
	int mCenterNumber;

	void TextureDetected(std::map< short, std::pair< int, std::string > >& objects_counters, IplImage* original, bool show_garbage);
	double cvMatchShapesNew(CvHuMoments* HuMoments1, CvHuMoments* HuMoments2, int method);
	RoadSigns* Object(MouseClick& my_mouse, short idx, std::string& object_name);
};

