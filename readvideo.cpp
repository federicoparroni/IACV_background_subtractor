#include <iostream>
#include <opencv2/opencv.hpp>
#include "pbas.cpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    VideoCapture cap("videos/camera1.mp4");
    PBAS *pbas = new PBAS();

    Mat frame;
    Mat gray;
    Mat *mask;

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    while(1){
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        mask = pbas->process(&gray);
        imshow("Frame", gray);
        imshow("Mask", *mask);
        char c=(char)waitKey(25);
        if(c==27) break;
    }
    pbas->~PBAS();
    cap.release();
    destroyAllWindows();
    return 0;
}  
