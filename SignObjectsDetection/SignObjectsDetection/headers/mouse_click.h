#pragma once

#include <cv.h>
#include <cxcore.h>

#ifndef _EiC
#include "cv.h"
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

// Point in textural features area
class Point {
public:
    Point(double _hu0,
          double _hu1,
          double _hu2,
          double _hu3,
          double _hu4,
          double _hu5);
    double mHu0;
    double mHu1;
    double mHu2;
    double mHu3;
    double mHu4;
    double mHu5;
};

// Mouse click handler
class MouseClick
{
public:
    MouseClick(bool filter_image, int width, int height);
    void SetImage(IplImage* image);
    void SetCrackImage(IplImage* image, const std::string& object_name);
    static void MyMouseClick(int event, int x, int y, int flags, void* param);
    static void MyMouseClickForTrain(int event, int x, int y, int flags, void* param);

    int mColors[3], mWidth, mHeight, mX, mY;
    IplImage* mImage;
    IplImage* mImageCrack;
    std::string mObjectName;
    bool mFilterImage;
    //CvHuMoments* mPoint;
    Point* mPoint;
};
