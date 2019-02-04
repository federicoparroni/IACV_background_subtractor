#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "pbas.cpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
    string filename = "videos/camera1.mp4";
    if(argc >= 2) filename = argv[1];
    int N = 30;
    int K = 3;
    float R_incdec = 0.05;
    int R_lower = 18;
    int R_scale = 5;
    float T_dec = 0.05;
    int T_inc = 1;
    int T_lower = 2;
    int T_upper = 200;
    if(argc == 11) {
        N = stoi(argv[2]);
        K = stoi(argv[3]);
        R_incdec = stof(argv[4]);
        R_lower = stoi(argv[5]);
        R_scale = stoi(argv[6]);
        T_dec = stof(argv[7]);
        T_inc = stoi(argv[8]);
        T_lower = stoi(argv[9]);
        T_upper = stoi(argv[10]);
    }
    VideoCapture cap(filename);
    PBAS *pbas = new PBAS(N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper);

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
