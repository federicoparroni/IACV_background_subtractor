#include <iostream>
#include <opencv2/opencv.hpp>
#include "pbas.cpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    VideoCapture cap("videos/salitona.mp4");
    PBAS *pbas = new PBAS(20);

    Mat frame;
    Mat gray;
    Mat *mask;

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    while(1){
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        mask = pbas->process(&gray);
        imshow("Frame", *mask);
        char c=(char)waitKey(25);
        if(c==27) break;
    }
    cap.release();
    destroyAllWindows(); 
    return 0;
}  
