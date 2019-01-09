#include "auto_correct.h"
#include <iostream>

AutoCorrect::AutoCorrect(IplImage* original)
{
    mOriginal = original;
    mChannel1 = cvCreateImage(cvGetSize(mOriginal), 8, 1);
    mChannel2 = cvCreateImage(cvGetSize(mOriginal), 8, 1);
    mChannel3 = cvCreateImage(cvGetSize(mOriginal), 8, 1);
    mAfterCorrect = cvCreateImage(cvGetSize(mOriginal), 8, 3);
}

void AutoCorrect::AutoLevels()
{
    cvSplit(mOriginal, mChannel1, mChannel2, mChannel3, NULL);
    cvNormalize(mChannel1, mChannel1, 0.0, 255.0, CV_MINMAX, NULL);
    cvNormalize(mChannel2, mChannel2, 0.0, 255.0, CV_MINMAX, NULL);
    cvNormalize(mChannel3, mChannel3, 0.0, 255.0, CV_MINMAX, NULL);
}

void AutoCorrect::GrayWorld()
{
    CvScalar av = cvAvg(mOriginal, NULL);
    double min_val = 0.0;
    double max_val = 0.0;
    CvPoint minloc, maxloc;
    IplImage* res = cvCreateImage(cvGetSize(mOriginal), 8, 1);
    cvCvtColor(mOriginal, res, CV_BGR2GRAY);
    cvMinMaxLoc(res, &min_val, &max_val, &minloc, &maxloc, 0);
    double f = (max_val + min_val) / 2;

    av.val[0] = f / av.val[0];
    av.val[1] = f / av.val[1];
    av.val[2] = f / av.val[2];

    cvConvertScale(mChannel1, mChannel1, av.val[0], 0.0);
    cvConvertScale(mChannel2, mChannel2, av.val[1], 0.0);
    cvConvertScale(mChannel3, mChannel3, av.val[2], 0.0);
    cvMerge(mChannel1, mChannel2, mChannel3, NULL, mAfterCorrect);
    cvReleaseImage(&res);
}

IplImage* AutoCorrect::GetResult()
{
    AutoLevels();
    GrayWorld();

    cvReleaseImage(&mChannel1);
    cvReleaseImage(&mChannel2);
    cvReleaseImage(&mChannel3);
    return mAfterCorrect;
}
