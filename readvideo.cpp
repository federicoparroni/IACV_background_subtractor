#include <iostream>
#include <opencv2/opencv.hpp>
#include "pbas.cpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    VideoCapture cap("videos/salitona.mp4");
    PBAS *pbas = new PBAS();

    Mat frame;
    Mat* mask;

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    while(1){
        cap >> frame;
        if (frame.empty()) break;

        mask = pbas->process(&frame);
        imshow("Frame", frame);
        moveWindow("Frame", 120,20);
        imshow("Mask", *mask);
        moveWindow("Mask", 490,20);
        char c=(char)waitKey(25);
        if(c==27) break;
    }
    pbas->~PBAS();
    cap.release();
    destroyAllWindows();
    return 0;
}  
