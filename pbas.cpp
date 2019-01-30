#include <iostream>
#include <math.h>
#include "pbas.h"
#include <stdlib.h>
#include <time.h>
#include <utility>      
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <chrono>
using namespace std::chrono;

using namespace std;


PBAS::PBAS() {
    N = 30;
    K = 4;
    R_incdec = 0.05;
    R_lower = 18;
    R_scale = 5;
    T_dec = 0.05;
    T_inc = 1;
    T_lower = 2;
    T_upper = 200;
    alpha = 10;
    I_m = 1.0;
    init();
}
PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200, int alpha = 10)
{
    this->N = N;
    this->K = K;
    this->R_incdec = R_incdec;
    this->R_lower = R_lower;
    this->R_scale = R_scale;
    this->T_dec = T_dec;
    this->T_inc = T_inc;
    this->T_lower = T_lower;
    this->T_upper = T_upper;
    this->alpha = alpha;
    this->I_m = 1.0;
    init();
}

void PBAS::init() {
    // initialize displacement array
    displacement_vec.push_back(make_pair(-1, -1));
    displacement_vec.push_back(make_pair(-1, 1));
    displacement_vec.push_back(make_pair(-1, 0));
    displacement_vec.push_back(make_pair(1, -1));
    displacement_vec.push_back(make_pair(1, 1));
    displacement_vec.push_back(make_pair(1, 0));
    displacement_vec.push_back(make_pair(0, 1));
    displacement_vec.push_back(make_pair(0, -1));
}

PBAS::~PBAS() {
    frame.release();
    frame_grad.release();
    median.release();
    B.clear();
    R.release();
    D.clear();
    T.release();
    F.release();
    d_minavg.release();
    displacement_vec.clear();
}

float PBAS::distance(uint8_t a, uint8_t b) {
    return abs(a-b);
}

float PBAS::distance(uint8_t p, uint8_t p_grad, uint8_t g, uint8_t g_grad) {
    return (this->alpha/this->I_m) * abs(p - g) + abs(p_grad - g_grad); 
}

Mat PBAS::gradient_magnitude(Mat* frame){
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Gradient X
    Sobel(*frame, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    // Gradient Y
    Sobel(*frame, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    // Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}

void PBAS::init_Mat(Mat* matrix, float initial_value){
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            matrix->at<float>(i,j) = initial_value;
        }
    }
}

Mat* PBAS::process(const Mat* frame) {
    //convert the frame in rgb and store it in the class variable this->frame
    cvtColor(*frame, this->frame, cv::COLOR_RGB2GRAY);
    //assign to the class variable the rgb frame
    this->frame_rgb = *frame;
    this->w = this->frame.cols;
    this->h = this->frame.rows;

    // gradients computation
    this->frame_grad = gradient_magnitude(&this->frame);
    this->I_m = mean(this->frame_grad).val[0];

    // B, D, d_minavg initialization
    if (B.size() == 0) {
        for(int i=0; i<N; i++) {
            Mat b_elem(h, w, CV_8UC1);
            randu(b_elem, 0, 255);
            B.push_back(b_elem);

            Mat b_grad_elem(h, w, CV_8UC1);
            randu(b_grad_elem, 0, 255);
            B_grad.push_back(b_grad_elem);

            Mat d_elem = Mat::zeros(h, w, CV_8UC1);
            D.push_back(d_elem);
        }
        F = Mat::zeros(h, w, CV_8UC1);
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        T = Mat::zeros(h, w, CV_32FC1);
        R = Mat::zeros(h, w, CV_32FC1);
        
        //initialize the median with the first frame
        median = this->frame.clone();

        init_Mat(&T, T_lower);
        init_Mat(&R, R_lower);
    }

    int channels = this->frame.channels();
    int nRows = this->h;
    int nCols = w * channels;
    // int y;
    // if (frame.isContinuous() && F.isContinuous() && R.isContinuous() && T.isContinuous()) {
    //     nCols *= nRows;
    //     nRows = 1;
    // }
    auto start = high_resolution_clock::now();
    for(int x=0; x < nRows; ++x) {
        this->i = this->frame.ptr<uint8_t>(x);
        this->i_grad = frame_grad.ptr<uint8_t>(x);
        this->q = F.ptr<uint8_t>(x);
        this->r = R.ptr<float>(x);
        this->t = T.ptr<float>(x);
        this->med = median.ptr<uint8_t>(x);
        this->i_rgb = this->frame_rgb.ptr<uint8_t>(x);

        for (int i_ptr=0; i_ptr < nCols; ++i_ptr) {
            //y = i_ptr % (channels * this->h);
            updateMedian(i_ptr);
            updateF(x, i_ptr, i_ptr);
            updateT(x, i_ptr, i_ptr);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << "ms" << endl;
    
    return &F;
    // return &median;
}

void PBAS::updateF(int x, int y, int i_ptr) {
    //c = 0
    //while c < 3 or k >= self.K:
    int k = 0;       // number of lower-than-R distances for the channel 'c'
    int j = 0;
    while(j < N && k < K) {
        //if(distance(i[i_ptr], i_grad[i_ptr], B[j].at<uint8_t>(x,y), B_grad[j].at<uint8_t>(x,y)) < r[i_ptr]) {
        if(distance(i[i_ptr], B[j].at<uint8_t>(x,y)) < r[i_ptr]) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y) => background pixel
    if(k >= K) {
        q[i_ptr] = 0;
        updateB(x, y, i_ptr);
    } else {
        //is_shadow(i_ptr);
        q[i_ptr] = 255;
    }
}

void PBAS::updateB(int x, int y, int i_ptr) {
    int rand_numb, n, y_disp, x_disp;
    pair<int, int> disp;
    float update_p;

    //initialize random seed
    srand (time(NULL));
    // generate a number between 0 and 99
    rand_numb = rand() %100;
    // get the T[x,y]
    update_p = 100 / t[i_ptr];
    n = rand() % N;

    if(rand_numb < update_p) {
        //generate a random number between 0 and N-1
        B[n].at<uint8_t>(x, y) = i[i_ptr];
        B_grad[n].at<uint8_t>(x, y) = i_grad[i_ptr];
        updateR(x, y, n, i_ptr);
    }
    
    rand_numb = rand() %100;
    if(rand_numb < update_p){
        //generate a random number between 0 and N-1
        y_disp = 0;
        x_disp = 0;

        while((x_disp == 0 && y_disp == 0) || x+x_disp >= h || y+y_disp >= w || x+x_disp < 0 || y+y_disp < 0){
            rand_numb = rand() %8;
            disp = displacement_vec[rand_numb];
            x_disp = disp.first;
            y_disp = disp.second;
        }

        B[n].at<uint8_t>(x+x_disp, y+y_disp) = frame.at<uint8_t>(x+x_disp, y+y_disp);
        
        updateR_notoptimized(x+x_disp, y+y_disp, n);
    }
}

void PBAS::updateR(int x, int y, int n, int i_ptr) {
    // find dmin
    uint8_t I = i[i_ptr];
    uint8_t I_grad = i_grad[i_ptr];
    int d_min = 255;
    int d_act = 0;
    for (int i=0; i<N; i++){
        //d_act = distance(I, I_grad, B[i].at<uint8_t>(x, y), B_grad[i].at<uint8_t>(x, y));
        d_act = distance(I, B[i].at<uint8_t>(x, y));
        if (d_act < d_min)
            d_min = d_act;
    }

    // update Dk
    D[n].at<uint8_t>(x, y) = d_min;

    // find davg
    unsigned int d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += (int)D[i].at<uint8_t>(x, y);
    }
    d_minavg.at<float>(x,y) = d_cum/float(N);

    // update R
    if (r[i_ptr] > d_minavg.at<float>(x,y) * R_scale){
        r[i_ptr] = r[i_ptr] * (1 - R_incdec);
    } else {
        r[i_ptr] = r[i_ptr] * (1 + R_incdec);
    }
    //cant got under R_lower
    r[i_ptr] = max((float)R_lower, r[i_ptr]);
}

void PBAS::updateR_notoptimized(int x, int y, int n) {
    // find dmin
    uint8_t I = frame.at<uint8_t>(x, y);
    uint8_t I_grad = frame_grad.at<uint8_t>(x,y);
    int d_min = 255;
    int d_act = 0;
    for (int i=0; i<N; i++){
        //d_act = distance(I, I_grad, B[i].at<uint8_t>(x, y), B_grad[i].at<uint8_t>(x, y));
        d_act = distance(I, B[i].at<uint8_t>(x, y));
        if (d_act < d_min)
            d_min = d_act;
    }

    // update Dk
    D[n].at<uchar>(x, y) = d_min;

    // find davg
    int d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += (int)D[i].at<uchar>(x, y);
    }
    d_minavg.at<float>(x,y) = d_cum/float(N);

    // update R
    if (R.at<float>(x,y) > d_minavg.at<float>(x,y) * R_scale){
        R.at<float>(x,y) = R.at<float>(x,y)*(1 - R_incdec);
    } else {
        R.at<float>(x,y) = R.at<float>(x,y)*(1 + R_incdec);
    }
    //cant got under R_lower
    R.at<float>(x,y) = max((float)R_lower, R.at<float>(x,y));
}

void PBAS::updateT(int x, int y, int i_ptr) {
    float Tinc_over_dmin;
    Tinc_over_dmin = T_inc / d_minavg.at<float>(x,y);
    if(q[i_ptr] == 255)
        t[i_ptr] += Tinc_over_dmin;
    else
        t[i_ptr] -= Tinc_over_dmin;
    t[i_ptr] = max((float)T_lower, t[i_ptr]);
    t[i_ptr] = min((float)T_upper, t[i_ptr]);
}

void PBAS::updateMedian(int col){
    uint8_t med_pixel = this->med[col];
    uint8_t frame_pixel = this->i[col];
    if(med_pixel != frame_pixel){
        if(med_pixel > frame_pixel) {
            this->med[col]--;
        } else {
            this->med[col]++;
        }
    }
}

void PBAS::is_shadow(int col){
    Mat i_hsv(Size(1,1),CV_8UC1);
    Mat m_hsv(Size(1,1),CV_8UC1);

    cvtColor(this->i_rgb[col], i_hsv, cv::COLOR_RGB2HSV);
    cvtColor(this->med[col], m_hsv, cv::COLOR_RGB2HSV);
    int a=4;
}



