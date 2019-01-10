#include "detection_objects.h"
#include "mouse_click.h"
#include <iostream>
#include <math.h>
#include <future>
#include "ap.h"
#include "dataanalysis.h"
#include "thread_pool.h"

using namespace alglib;

#define SQUARE_LIMIT 15 * 50
#define PERIMETER_LIMIT 100


BaseObject::BaseObject(const MouseClick& my_mouse)
{
    mRed = my_mouse.mColors[2];
    mGreen = my_mouse.mColors[1];
    mBlue = my_mouse.mColors[0];
    mX = my_mouse.mX;
    mY = my_mouse.mY;
}


RoadSigns::RoadSigns(const std::string& object_name, int width, int height)
{
    mObjectName = object_name;
    mWidth = width;
    mHeight = height;
    mCount = 0;
}

bool RoadSigns::PointInEpsilon(uchar* ptr, int& epsilon, int& x)
{
    bool exist = false;
    for (auto it = mOptions.begin(); it != mOptions.end(); it++)
    {
        if (exist)
            break;

        BaseObject* obj = *it;
        exist = ((ptr[3 * x] - obj->mBlue) * (ptr[3 * x] - obj->mBlue) +
            (ptr[3 * x + 1] - obj->mGreen) * (ptr[3 * x + 1] - obj->mGreen) +
            (ptr[3 * x + 2] - obj->mRed) * (ptr[3 * x + 2] - obj->mRed) <= epsilon * epsilon);
    }

    return exist;
}

void RoadSigns::Inc()
{
    mCount++;
}

void RoadSigns::Clear()
{
    mOptions.clear();
}


ObjectsDetection::ObjectsDetection(int epsilon, int width, int height)
{
    mEpsilon = epsilon;
    mWidth = width;
    mHeight = height;
}

void ObjectsDetection::SetImage(IplImage* image)
{
    mColorImage = image;
}

bool ObjectsDetection::ColorDetected(RoadSigns* road_sign, int height, int width, int eps, IplImage* img)
{
    for (int y = height - 1; y >= 0; --y)
    {
        uchar* ptr = (uchar*)(img->imageData + y * img->widthStep);
        for (int x = width - 1; x >= 0; --x)
        {
            if (road_sign->PointInEpsilon(ptr, eps, x))
                ptr[3 * x] = ptr[3 * x + 1] = ptr[3 * x + 2] = 255;
        }
    }

    return true;
}

void ObjectsDetection::ColorDetectedMass()
{
    //TODO: my thread pool don't work on Linux
    //using FutureShared = std::shared_ptr<FutureObject<bool>>;
    using Future = std::future<bool>;

    std::vector<Future> task_list;
    size_t idx = 0;
    for (auto sign_it = mSignsList.begin(); sign_it != mSignsList.end(); sign_it++)
    {
        if (!(*sign_it).second)
        {
            idx++;
            continue;
        }

        if (idx == mSignsList.size() - 1)
            ColorDetected((*sign_it).second, mHeight, mWidth, mEpsilon, mColorImage);
        else
        {
            //ThreadPool pool;
            //auto task = pool.RunAsync<bool>(&ColorDetected, (*sign_it).second, mHeight, mWidth, mEpsilon, mColorImage);
            auto task = std::async(&ColorDetected, (*sign_it).second, mHeight, mWidth, mEpsilon, mColorImage);
            task_list.push_back(std::move(task));
        }

        idx++;
    }

    std::for_each(task_list.begin(), task_list.end(), [](Future& task) {
        //while (!task->finished);
        task.get();
    });

    for (int y = mHeight - 1; y >= 0; --y)
    {
        uchar* ptr = (uchar*)(mColorImage->imageData + y * mColorImage->widthStep);
        for (int x = mWidth - 1; x >= 0; --x)
        {
            if (ptr[3 * x] != 255 || ptr[3 * x + 1] != 255 || ptr[3 * x + 2] != 255)
                ptr[3 * x] = ptr[3 * x + 1] = ptr[3 * x + 2] = 0;
        }
    }
}

void ObjectsDetection::TextureDetected(std::map< short, std::pair< int, std::string > >& objects_counters, IplImage* original, bool show_garbage)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = 0;

    IplImage* gray = cvCreateImage(cvGetSize(original), IPL_DEPTH_8U, 1);
    cvCvtColor(mColorImage, gray, CV_RGB2GRAY);
    int contoursCont = cvFindContours(gray, storage, &contours, sizeof(CvContour));
    for (CvSeq* seq0 = contours; seq0 != 0; seq0 = seq0->h_next)
    {
        CvSeq* result = seq0;
        double area = fabs(cvContourArea(result));
        double perim = cvContourPerimeter(result);
        if (area < SQUARE_LIMIT || perim < PERIMETER_LIMIT)
            continue;

        CvMoments moments;
        CvHuMoments hu_moments;
        cvMoments(result, &moments, 0);
        cvGetHuMoments(&moments, &hu_moments);

        real_1d_array inputArrayCheck;
        inputArrayCheck.setlength(mNumberOfFeatures);
        real_1d_array outputArrayCheck;
        outputArrayCheck.setlength(mCenterNumber);

        inputArrayCheck[FIRST_MOMENT] = hu_moments.hu1;
        inputArrayCheck[SECOND_MOMENT] = hu_moments.hu2;
        inputArrayCheck[THIRD_MOMENT] = hu_moments.hu3;
        inputArrayCheck[FORTH_MOMENT] = hu_moments.hu4;
        inputArrayCheck[FIFTH_MOMENT] = hu_moments.hu5;
        inputArrayCheck[SIXTH_MOMENT] = hu_moments.hu6;
        //inputArrayCheck[SEVENTH_MOMENT] = hu_moments.hu7;

        mlpprocess(mNet, inputArrayCheck, outputArrayCheck);

        float max_out_prob = outputArrayCheck[0];
        int object_idx = 0;
        for (int check = 1; check < mCenterNumber; ++check)
            if (outputArrayCheck[check] > max_out_prob)
            {
                max_out_prob = outputArrayCheck[check];
                object_idx = check;
            }

        RoadSigns* object;
        if (mSignsList.count(object_idx))
            object = mSignsList[object_idx];

        if (!object || (object_idx == GARBAGE && !show_garbage))
            continue;

        if (objects_counters.count(object_idx) == 0)
            objects_counters[object_idx] = std::make_pair(0, object->Name());

        objects_counters[object_idx].first++;
        CvBox2D rect = cvMinAreaRect2(seq0, storage);
        CvPoint pt = cvPoint(rect.center.x, rect.center.y);
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 1, CV_AA);
        cvPutText(original, object->Name().c_str(), pt, &font, CV_RGB(150, 0, 150));
        cvDrawContours(original, seq0, CV_RGB(255, 0, 0), CV_RGB(100, 100, 100), 0, 3, 8);
    }
    cvReleaseImage(&gray);
}

void ObjectsDetection::Detected(IplImage* original, bool show_garbage)
{
    ColorDetectedMass();
    std::map< short, std::pair< int, std::string > > objects_counters;
    TextureDetected(objects_counters, original, show_garbage);

    for (auto sign_it = mSignsList.begin(); sign_it != mSignsList.end(); sign_it++)
    {
        short idx = (*sign_it).first;
        RoadSigns* road_signs = (*sign_it).second;
        if (!objects_counters.count(idx))
            continue;

        std::pair< int, std::string > obj_map = objects_counters[idx];
        if (obj_map.first < THRESHOLD)
            continue;

        //std::cout << obj_map.second << " detected" << "\n";
        road_signs->Inc();
    }
}

RoadSigns* ObjectsDetection::Object(MouseClick& my_mouse, short idx, std::string& object_name)
{
    if (!my_mouse.mFilterImage)
        return nullptr;

    RoadSigns* road_obj;
    if (mSignsList.count(idx) == 0)
    {
        RoadSigns* new_obj = new RoadSigns(object_name, my_mouse.mWidth, my_mouse.mHeight);
        mSignsList[idx] = new_obj;
        road_obj = new_obj;
    }
    else
        road_obj = mSignsList[idx];

    return road_obj;
}

void ObjectsDetection::AddOptionsToObject(MouseClick& my_mouse, short idx, std::string& object_name)
{
    RoadSigns* road_obj = Object(my_mouse, idx, object_name);
    if (!road_obj)
        return;

    BaseObject* opt_obj = new BaseObject(my_mouse);
    road_obj->mOptions.push_back(opt_obj);
    my_mouse.mFilterImage = false;
}

void ObjectsDetection::AddTextureToObject(MouseClick& my_mouse, short idx, std::string& object_name, const short& mode)
{
    RoadSigns* road_obj = Object(my_mouse, idx, object_name);
    if (!road_obj)
        return;

    if (mode == 1)
        road_obj->mTexturesBase.push_back(my_mouse.mPoint);
    else if (mode == 2)
        road_obj->mTexturesTrain.push_back(my_mouse.mPoint);
    my_mouse.mFilterImage = false;
}

double ObjectsDetection::cvMatchShapesNew(CvHuMoments* HuMoments1, CvHuMoments* HuMoments2, int method)
{
    double ma[7], mb[7];
    int i, sma, smb;
    double eps = 1.e-5;
    double mmm;
    double result = 0;
    ma[0] = HuMoments1->hu1;
    ma[1] = HuMoments1->hu2;
    ma[2] = HuMoments1->hu3;
    ma[3] = HuMoments1->hu4;
    ma[4] = HuMoments1->hu5;
    ma[5] = HuMoments1->hu6;
    ma[6] = HuMoments1->hu7;
    mb[0] = HuMoments2->hu1;
    mb[1] = HuMoments2->hu2;
    mb[2] = HuMoments2->hu3;
    mb[3] = HuMoments2->hu4;
    mb[4] = HuMoments2->hu5;
    mb[5] = HuMoments2->hu6;
    mb[6] = HuMoments2->hu7;
    switch (method)
    {
    case 1:
    {
        for (i = 0; i < 7; i++)
        {
            double ama = fabs(ma[i]);
            double amb = fabs(mb[i]);
            if (ma[i] > 0)
                sma = 1;
            else if (ma[i] < 0)
                sma = -1;
            else
                sma = 0;
            if (mb[i] > 0)
                smb = 1;
            else if (mb[i] < 0)
                smb = -1;
            else
                smb = 0;
            if (ama > eps && amb > eps)
            {
                ama = 1. / (sma * log10(ama));
                amb = 1. / (smb * log10(amb));
                result += fabs(-ama + amb);
            }
        }
        break;
    }
    case 2:
    {
        for (i = 0; i < 7; i++)
        {
            double ama = fabs(ma[i]);
            double amb = fabs(mb[i]);
            if (ma[i] > 0)
                sma = 1;
            else if (ma[i] < 0)
                sma = -1;
            else
                sma = 0;
            if (mb[i] > 0)
                smb = 1;
            else if (mb[i] < 0)
                smb = -1;
            else
                smb = 0;
            if (ama > eps && amb > eps)
            {
                ama = sma * log10(ama);
                amb = smb * log10(amb);
                result += fabs(-ama + amb);
            }
        }
        break;
    }
    case 3:
    {
        for (i = 0; i < 7; i++)
        {
            double ama = fabs(ma[i]);
            double amb = fabs(mb[i]);
            if (ma[i] > 0)
                sma = 1;
            else if (ma[i] < 0)
                sma = -1;
            else
                sma = 0;
            if (mb[i] > 0)
                smb = 1;
            else if (mb[i] < 0)
                smb = -1;
            else
                smb = 0;
            if (ama > eps && amb > eps)
            {
                ama = sma * log10(ama);
                amb = smb * log10(amb);
                mmm = fabs((ama - amb) / ama);
                if (result < mmm)
                    result = mmm;
            }
        }
        break;
    }
    default:
        CV_Error(CV_StsBadArg, "Unknown comparison method");
    }
    return result;
}

void ObjectsDetection::TrainNet()
{
    mNumberOfFeatures = LAST_MOMENT;
    int trainDataSize = 0;
    std::vector< Point* > all_textures;
    std::vector< Point* > main_textures;
    std::map< int, int > index_map;
    for (auto sign_it = mSignsList.begin(); sign_it != mSignsList.end(); sign_it++)
    {
        if (!(*sign_it).second)
            continue;

        trainDataSize += (*sign_it).second->mTexturesTrain.size();
        std::vector< Point* >& textures = (*sign_it).second->mTexturesTrain;
        all_textures.reserve(all_textures.size() + textures.size());
        all_textures.insert(all_textures.end(), textures.begin(), textures.end());

        std::vector< Point* >& base = (*sign_it).second->mTexturesBase;
        for (int idx = 0; idx < base.size(); idx++)
        {
            Point* tmp_point = base[idx];
            if (!tmp_point)
                continue;

            index_map[main_textures.size()] = (*sign_it).first;
            main_textures.push_back(tmp_point);
        }
        //main_textures.reserve( main_textures.size() + base.size() );
        //main_textures.insert( main_textures.end(), base.begin(), base.end() );
    }

    float** trainData = new float*[trainDataSize];
    for (int i = 0; i < trainDataSize; ++i)
        trainData[i] = new float[mNumberOfFeatures + 1];

    int counter = 0;
    for (auto point_it = all_textures.begin(); point_it != all_textures.end(); point_it++)
    {
        Point* one = *point_it;
        std::vector< double > distance;
        for (auto main = main_textures.begin(); main != main_textures.end(); main++)
        {
            double dist = sqrt((double)(one->mHu0 - (*main)->mHu0) * (one->mHu0 - (*main)->mHu0) +
                (one->mHu1 - (*main)->mHu1) * (one->mHu1 - (*main)->mHu1) +
                (one->mHu2 - (*main)->mHu2) * (one->mHu2 - (*main)->mHu2) +
                (one->mHu3 - (*main)->mHu3) * (one->mHu3 - (*main)->mHu3) +
                (one->mHu4 - (*main)->mHu4) * (one->mHu4 - (*main)->mHu4) +
                (one->mHu5 - (*main)->mHu5) * (one->mHu5 - (*main)->mHu5));
            distance.push_back(dist);
            //double dist = cvMatchShapesNew( one, ( *main ), 3 );
            distance.push_back(dist);
        }

        double mindist = distance[0];
        int nummin = 0;
        for (int d = 1; d < distance.size(); d++)
        {
            if (distance[d] > mindist)
                continue;

            mindist = distance[d];
            nummin = index_map[d];
        }

        trainData[counter][FIRST_MOMENT] = one->mHu0;
        trainData[counter][SECOND_MOMENT] = one->mHu1;
        trainData[counter][THIRD_MOMENT] = one->mHu2;
        trainData[counter][FORTH_MOMENT] = one->mHu3;
        trainData[counter][FIFTH_MOMENT] = one->mHu4;
        trainData[counter][SIXTH_MOMENT] = one->mHu5;
        //trainData[counter][SEVENTH_MOMENT] = one->hu7;
        trainData[counter][mNumberOfFeatures] = nummin;
        counter++;
    }

    real_2d_array trainData_f;
    trainData_f.setlength(trainDataSize, mNumberOfFeatures + 1);
    for (int i = 0; i < trainDataSize; ++i)
        for (int j = 0; j < mNumberOfFeatures + 1; ++j)
            trainData_f[i][j] = trainData[i][j];

    int layer1 = 10;
    int layer2 = 5;
    mCenterNumber = main_textures.size();
    mlpcreatec2(mNumberOfFeatures, layer1, layer2, mCenterNumber, mNet);
    mlpreport report;
    ae_int_t info;
    mlptrainlm(mNet, trainData_f, trainDataSize, 0.001, 2, info, report);
}

void ObjectsDetection::ShowContours(IplImage* original)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = 0;

    IplImage* gray = cvCreateImage(cvGetSize(original), IPL_DEPTH_8U, 1);
    cvCvtColor(mColorImage, gray, CV_RGB2GRAY);
    int contoursCont = cvFindContours(gray, storage, &contours, sizeof(CvContour));
    for (CvSeq* seq0 = contours; seq0 != 0; seq0 = seq0->h_next)
    {
        double area = fabs(cvContourArea(seq0));
        double perim = cvContourPerimeter(seq0);
        if (area < SQUARE_LIMIT || perim < PERIMETER_LIMIT)
            continue;

        cvDrawContours(original, seq0, CV_RGB(255, 0, 0), CV_RGB(100, 100, 100), 0, 3, 8);
    }

    cvReleaseMemStorage(&storage);
    cvReleaseImage(&gray);
}

void ObjectsDetection::ClearOptions()
{
    for (auto sign_it = mSignsList.begin(); sign_it != mSignsList.end(); sign_it++)
    {
        if (!(*sign_it).second)
            continue;

        (*sign_it).second->Clear();
    }
}

void ObjectsDetection::ClearTextures()
{
    for (auto sign_it = mSignsList.begin(); sign_it != mSignsList.end(); sign_it++)
    {
        if (!(*sign_it).second)
            continue;

        (*sign_it).second->mTexturesTrain.clear();
        (*sign_it).second->mTexturesBase.clear();
    }
}
