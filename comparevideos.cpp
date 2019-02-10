#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/stat.h> 
#include "pbas_rbg.cpp"
//#include <limits>

using namespace std;
using namespace cv;

int current_frame = 0;
float FPS = 30;

int main(int argc, char const *argv[]) {
    // init
    int N = 30;
    int K = 3;
    float R_incdec = 0.05;
    int R_lower = 18;
    int R_scale = 5;
    float T_dec = 0.05;
    int T_inc = 2;
    int T_lower = 2;
    int T_upper = 200;
    int alpha = 10;
    if(argc == 11) {
        N = stoi(argv[1]);
        K = stoi(argv[2]);
        R_incdec = stof(argv[3]);
        R_lower = stoi(argv[4]);
        R_scale = stoi(argv[5]);
        T_dec = stof(argv[6]);
        T_inc = stoi(argv[7]);
        T_lower = stoi(argv[8]);
        T_upper = stoi(argv[9]);
        alpha = stoi(argv[10]);
    }
    VideoCapture cap1("videos/railway_s.mp4");
    //VideoCapture cap2("videos/nighttraffic.mp4");
    //VideoCapture cap3("videos/railway_s.mp4");

    PBAS *pbas1 = new PBAS(N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha);
    //PBAS *pbas2 = new PBAS(N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha);
    //PBAS *pbas3 = new PBAS(N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha);
    pbas1->verbose = false; // pbas2->verbose = false; pbas3->verbose = false;
    Mat frame1, frame2, frame3;
    Mat mask1, mask2, mask3;
    
    // process
    while(1) {
        //cout << current_frame / FPS << "s" << endl;
        cap1 >> frame1;
        //cap2 >> frame2;
        //cap3 >> frame3;
        //if (frame1.empty() || frame2.empty() || frame3.empty()) break;

        //GaussianBlur(frame1, frame1, Size(3,3), 0,0);
        //GaussianBlur(frame2, frame2, Size(3,3), 0,0);
        //GaussianBlur(frame3, frame3, Size(3,3), 0,0);

        mask1 = pbas1->process(frame1);
        //mask2 = pbas2->process(frame2);
        //mask3 = pbas3->process(frame3);
        
        imshow("Jackson1", frame1);
        if(current_frame == 0) moveWindow("Jackson1", 0,20);
        imshow("Jackson1 mask", mask1);
        if(current_frame == 0) moveWindow("Jackson1 mask", 420, 20);
/*
        imshow("Jackson2", frame2);
        if(current_frame == 0) moveWindow("Jackson2", 10,300);
        imshow("Jackson2 mask", mask2);
        if(current_frame == 0) moveWindow("Jackson2 mask", 410, 300);

        imshow("Railway", frame3);
        if(current_frame == 0) moveWindow("Railway", 10,580);
        imshow("Railway mask", mask3);
        if(current_frame == 0) moveWindow("Railway mask", 410, 580); */

        char c=(char)waitKey(25);
        if(c==27) break;

        current_frame ++;
    }
    pbas1->~PBAS(); //pbas2->~PBAS(); pbas3->~PBAS();
    cap1.release(); //cap2.release(); cap3.release();
    destroyAllWindows();
    return 0;
}
