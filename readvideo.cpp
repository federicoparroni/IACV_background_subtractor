#include <iostream>
#include <opencv2/opencv.hpp>
#include "pbas.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    char videopath[] = "/Users/federico/Documents/Repository/IACVproject/dataset/HighwayI/HighwayI_%06d.png";
    VideoCapture cap(videopath);

    PBAS *pbas = new PBAS();

    Mat frame;
    Mat gray;
    Mat *mask;
    while(1)
    {
        cap >> frame;
        if(!frame.data) break;

        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        //mask = pbas->process(&frame);
        imshow("Video", gray);

        if(waitKey(10) >= 0) break;              
    }

    return 0;
}
