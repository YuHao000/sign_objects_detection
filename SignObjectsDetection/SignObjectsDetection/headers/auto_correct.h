#pragma once

#include <cv.h>
#include <cxcore.h>

#ifndef _EiC
#include "cv.h"
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

// Auto correction class
class AutoCorrect
{
public:
    AutoCorrect(IplImage* original);
    IplImage* GetResult();
private:
    IplImage * mOriginal;
    IplImage* mAfterCorrect;
    IplImage* mChannel1;
    IplImage* mChannel2;
    IplImage* mChannel3;

    void AutoLevels();
    void GrayWorld();
};
