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

#include <memory>

#define THRESHOLD 5000

// All recognizable road signs
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

// Textural features: linear normalized moments
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

// Class for options storage
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


// Characteristics of road signs
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

using RoadSignsPtr = std::shared_ptr<RoadSigns>;

// Class to perform processing
class ObjectsDetection
{
public:
    ObjectsDetection(int epsilon, int width, int height);
    void TrainNet();
    void SetImage(IplImage* image);
    void ColorDetectedMass();
    static bool ColorDetected(RoadSignsPtr road_sign, int height, int width, int eps, IplImage* img);
    void Detected(IplImage* original, bool show_garbage);
    void AddOptionsToObject(MouseClick& my_mouse, short idx, std::string& object_name);
    void AddTextureToObject(MouseClick& my_mouse, short idx,
        std::string& object_name, const short& mode);
    void ShowContours(IplImage* original);
    void ClearOptions();
    void ClearTextures();
private:
    std::map< short, RoadSignsPtr > mSignsList;
    int mWidth;
    int mHeight;
    IplImage* mColorImage;
    int mEpsilon;

    alglib::multilayerperceptron mNet;
    int mNumberOfFeatures;
    int mCenterNumber;

    void TextureDetected(std::map< short, std::pair< int, std::string > >& objects_counters, IplImage* original, bool show_garbage);
    double cvMatchShapesNew(CvHuMoments* HuMoments1, CvHuMoments* HuMoments2, int method);
    RoadSignsPtr Object(MouseClick& my_mouse, short idx, std::string& object_name);
};

