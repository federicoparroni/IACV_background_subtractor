#include <iostream>
#include <opencv2/opencv.hpp>
#include "pbas.cpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    string filename = "videos/camera1.mp4";
    if(argc == 2) filename = argv[1];
    VideoCapture cap(filename);
    PBAS *pbas = new PBAS();

    Mat frame;
    Mat* mask;
    Mat edges;

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        //GaussianBlur(frame, frame, Size(3,3), 0, 0);

        mask = pbas->process(&frame);
        imshow("Frame", frame);
        moveWindow("Frame", 20,20);

        imshow("Mask", *mask);
        moveWindow("Mask", 500,20);

        // imshow("Shadows CNCC", pbas->shadow_cncc);
        // moveWindow("Shadows CNCC", 750,20);

        imshow("shadow_hsv", pbas->F_shadow_hsv);
        moveWindow("Shadows hsv", 20,400);

        // imshow("Shadows CNCC", pbas->shadow_cncc);
        // moveWindow("Shadows CNCC", 750,20);

        // imshow("shadow_hsv", pbas->F_shadow_hsv);
        // moveWindow("Shadows hsv", 350,400);

        // Mat converted;
        // pbas->median.convertTo(converted, CV_16SC3);
        // frame.convertTo(frame, CV_16SC3);
        // Mat res = abs(frame - converted);
        // res.convertTo(res, CV_8UC3);
        // cvtColor(res, res, COLOR_RGB2GRAY);
        // threshold(res, res, 50, 255, THRESH_BINARY);
        // imshow("bg model", res);
        // moveWindow("bg model", 350,400);

        char c=(char)waitKey(25);
        if(c==27) break;
    }
    pbas->~PBAS();
    cap.release();
    destroyAllWindows();
    return 0;
}  
