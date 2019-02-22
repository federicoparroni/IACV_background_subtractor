#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/stat.h> 
#include "pbas.cpp"

using namespace std;
using namespace cv;

int current_frame = 0;
float FPS = 30;

int main(int argc, char const *argv[]) {
    if(argc != 2) {
        cout << "Missing parameter: video path." << endl;
        exit(0);
    }
    string videopath = argv[1];
    VideoCapture cap(videopath);

    PBASgray *pbas = new PBASgray(30);

    Mat frame;
    Mat *mask;
    
    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        mask = pbas->process(&frame);

        imshow("Grayscale", frame);
        if(current_frame == 0) moveWindow("Grayscale", 0,20);
        imshow("Grayscale mask", *mask);
        if(current_frame == 0) moveWindow("Grayscale mask", 425, 20);

        char c=(char)waitKey(25);
        if(c==27) break;

        current_frame ++;
    }
    pbas->~PBASgray();
    cap.release();
    destroyAllWindows();
    return 0;
}
