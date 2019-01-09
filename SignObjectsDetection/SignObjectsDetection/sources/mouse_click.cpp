#include "mouse_click.h"
#include <iostream>

#define SQUARE_LIMIT 15 * 50
#define PERIMETER_LIMIT 100


Point::Point(double _hu0 = 0.0,
             double _hu1 = 0.0,
             double _hu2 = 0.0,
             double _hu3 = 0.0,
             double _hu4 = 0.0,
             double _hu5 = 0.0)
{
    mHu0 = _hu0;
    mHu1 = _hu1;
    mHu2 = _hu2;
    mHu3 = _hu3;
    mHu4 = _hu4;
    mHu5 = _hu5;
}

MouseClick::MouseClick(bool filter_image, int width, int height)
{
    mColors[0] = mColors[1] = mColors[2] = 0;
    mFilterImage = filter_image;
    mWidth = width;
    mHeight = height;
}

void MouseClick::SetImage(IplImage* image)
{
    mImage = image;
}

void MouseClick::SetCrackImage(IplImage* image, const std::string& object_name)
{
    mImageCrack = image;
    mObjectName = object_name;
}

void MouseClick::MyMouseClick(int event, int x, int y, int flags, void* param)
{
    if (event != 1)
        return;

    MouseClick* my_mouse = (MouseClick*)param;
    IplImage* tmp = my_mouse->mImage;
    uchar* ptr = (uchar*)(tmp->imageData + y * tmp->widthStep);
    my_mouse->mColors[0] = (ptr[3 * x]);
    my_mouse->mColors[1] = (ptr[3 * x + 1]);
    my_mouse->mColors[2] = (ptr[3 * x + 2]);
    my_mouse->mX = x;
    my_mouse->mY = y;
    my_mouse->mFilterImage = true;
}

void MouseClick::MyMouseClickForTrain(int event, int x, int y, int flags, void* param)
{
    if (event != 1)
        return;

    MouseClick* my_mouse = (MouseClick*)param;
    IplImage* tmp = my_mouse->mImage;
    IplImage* gray_tmp = cvCreateImage(cvGetSize(tmp), 8, 1);
    cvCvtColor(tmp, gray_tmp, CV_BGR2GRAY);

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = 0;
    CvSeq* sequence = 0;

    int contoursCont = cvFindContours(gray_tmp, storage, &contours, sizeof(CvContour));
    for (CvSeq* seq0 = contours; seq0 != 0; seq0 = seq0->h_next)
    {
        CvSeq* result = seq0;
        double area = fabs(cvContourArea(result));
        double perim = cvContourPerimeter(result);
        if (area < SQUARE_LIMIT || perim < PERIMETER_LIMIT)
            continue;

        CvPoint2D32f inside_contour;
        inside_contour.x = (float)x;
        inside_contour.y = (float)y;
        double checkPoint = cvPointPolygonTest(result, inside_contour, 0);
        if (checkPoint < 0)
            continue;

        CvMoments moments;
        CvHuMoments* hu_moments = new CvHuMoments();
        cvMoments(result, &moments, 0);
        cvGetHuMoments(&moments, hu_moments);
        my_mouse->mPoint = new Point(hu_moments->hu1,
            hu_moments->hu2,
            hu_moments->hu3,
            hu_moments->hu4,
            hu_moments->hu5,
            hu_moments->hu6);
        my_mouse->mFilterImage = true;
    }
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&gray_tmp);
}

