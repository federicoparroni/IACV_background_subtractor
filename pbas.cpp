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
using namespace cv;


PBAS::PBAS() {
    N = 5;
    K = 3;
    R_incdec = 0.05;
    R_lower = 18;
    R_scale = 5;
    T_dec = 0.05;
    T_inc = 1;
    T_lower = 2;
    T_upper = 200;
    alpha = 10;
    I_m = 1.0;
    ALPHA = 0.1;
    BETA = 0.9;
    TAU_H = 80;
    TAU_S = 20;
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
    return (this->alpha/this->I_m) * abs((int)p_grad - (int)g_grad) + abs((int)p - (int)g); 
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

Mat PBAS::shadows_corner(Mat* frame, Mat* mask){
    Mat dst, dst_norm, dst_norm_scaled, masked_frame;
    dst = Mat::zeros(frame->size(), CV_32FC1);
    /// Detector parameters
    int blockSize = 35;
    int apertureSize = 1;
    double k = 0.05;

    bitwise_and(*frame, *mask, masked_frame);
    /// Detecting corners
    cornerHarris(masked_frame, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    imshow("corners", dst_norm_scaled );

    dst_norm_scaled.setTo(255, dst_norm_scaled>=20);
    dst_norm_scaled.setTo(0, dst_norm_scaled<20);

    imshow("shadow_corners", dst_norm_scaled);

    return dst_norm_scaled;
}

//  Fast iteratation over Mat pixels: https://stackoverflow.com/a/46966298
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
            // Mat b_elem(h, w, CV_8UC1);
            // randu(b_elem, 0, 255);
            B.push_back(this->frame.clone());

            // Mat b_grad_elem(h, w, CV_8UC1);
            // randu(b_grad_elem, 0, 255);
            B_grad.push_back(frame_grad.clone());

            Mat d_elem = Mat::zeros(h, w, CV_8UC1);
            D.push_back(d_elem);
        }
        F = Mat::zeros(h, w, CV_8UC1);
        F_shadow_hsv = Mat::zeros(h, w, CV_8UC1);
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        T = Mat::zeros(h, w, CV_32FC1);
        R = Mat::zeros(h, w, CV_32FC1);
        
        //initialize the median with the first frame
        median = frame->clone();

        init_Mat(&T, T_lower);
        init_Mat(&R, R_lower);
        init_Mat(&d_minavg, 128);
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
        this->q_shadow_hsv = F_shadow_hsv.ptr<uint8_t>(x);
        this->r = R.ptr<float>(x);
        this->t = T.ptr<float>(x);
        this->med = median.ptr<Vec3b>(x);
        this->i_rgb = this->frame_rgb.ptr<Vec3b>(x);

        for (int i_ptr=0; i_ptr < nCols; ++i_ptr) {
            //y = i_ptr % (channels * this->h);
            updateMedian(i_ptr);
            updateF(x, i_ptr, i_ptr);
            updateT(x, i_ptr, i_ptr);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    medianBlur(F_shadow_hsv,F_shadow_hsv,9);
    cout << duration.count() << "ms" << endl;
    
    this->F = shadows_corner(&this->frame, &F);
    imshow("shadow_hsv", F_shadow_hsv);
    final_mask = F&F_shadow_hsv; 
    return &final_mask;
}

void PBAS::updateF(int x, int y, int i_ptr) {
    //c = 0
    //while c < 3 or k >= self.K:
    int k = 0;       // number of lower-than-R distances for the channel 'c'
    int j = 0;
    while(j < N && k < K) {
        if(distance(i[i_ptr], i_grad[i_ptr], B[j].at<uint8_t>(x,y), B_grad[j].at<uint8_t>(x,y)) < r[i_ptr]) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y) => background pixel
    if(k >= K) {
        q[i_ptr] = 0;
        q_shadow_hsv[i_ptr]=0;
        updateB(x, y, i_ptr);
    } else {
        if(!is_shadow(i_ptr)) q_shadow_hsv[i_ptr] = 255;
        else q_shadow_hsv[i_ptr] = 0;
        q[i_ptr] = 255;
    }
}

void PBAS::updateB(int x, int y, int i_ptr) {
    int rand_numb, n, y_disp, x_disp;
    pair<int, int> disp;
    float update_p;

    //initialize random seed
    //srand (time(NULL));
    // generate a number between 0 and 99
    rand_numb = rand() %100;
    // get the T[x,y]
    update_p = 100 / t[i_ptr];

    n = rand() % N;
    if(rand_numb < update_p) {
        //generate a random number between 0 and N-1
        
        updateR(x, y, n, i_ptr);
        B[n].at<uint8_t>(x, y) = i[i_ptr];
        B_grad[n].at<uint8_t>(x, y) = i_grad[i_ptr];
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
        updateR_notoptimized(x+x_disp, y+y_disp, n);
        B[n].at<uint8_t>(x+x_disp, y+y_disp) = frame.at<uint8_t>(x+x_disp, y+y_disp);
        B_grad[n].at<uint8_t>(x+x_disp, y+y_disp) = frame_grad.at<uint8_t>(x+x_disp, y+y_disp);
    }
}

void PBAS::updateR(int x, int y, int n, int i_ptr) {
    // find dmin
    uint8_t I = i[i_ptr];
    uint8_t I_grad = i_grad[i_ptr];
    int d_min = 255;
    int d_act = 0;
    for (int i=0; i<N; i++){
        d_act = distance(I, I_grad, B[i].at<uint8_t>(x, y), B_grad[i].at<uint8_t>(x, y));
        if (d_act < d_min)
            d_min = d_act;
    }

    // update Dk
    D[n].at<uint8_t>(x, y) = d_min;

    // find davg
    float d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += (int)D[i].at<uint8_t>(x, y);
    }
    d_minavg.at<float>(x,y) = d_cum/N;

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
        d_act = distance(I, I_grad, B[i].at<uint8_t>(x, y), B_grad[i].at<uint8_t>(x, y));
        if (d_act < d_min)
            d_min = d_act;
    }

    // update Dk
    D[n].at<uchar>(x, y) = d_min;

    // find davg
    float d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += (int)D[i].at<uchar>(x, y);
    }
    d_minavg.at<float>(x,y) = d_cum/N;
    // cout << d_minavg.at<float>(x,y) << endl;

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
    float oldT = t[i_ptr];
    if(q[i_ptr] == 255)
        t[i_ptr] += T_inc / (d_minavg.at<float>(x,y) + 1);
    else
        t[i_ptr] -= T_dec / (d_minavg.at<float>(x,y) + 1);
    //cout << d_minavg.at<float>(x,y) << endl;
    //cout << t[i_ptr] << endl; 

    t[i_ptr] = max((float)T_lower, t[i_ptr]);
    t[i_ptr] = min((float)T_upper, t[i_ptr]);
    // cout << "pos " + to_string(x) + " " + to_string(y) + ". old T: " + to_string(oldT) + " new T: " + to_string(t[i_ptr]) << endl;
}

void PBAS::updateMedian(int col){
    Vec3b med_pixel = this->med[col];
    Vec3b frame_pixel = this->i_rgb[col];

    for(uint8_t c=0; c < 3; c++) {    
        if(med_pixel[c] > frame_pixel[c]) {
            this->med[col][c]--;
        }
        if(med_pixel[c] < frame_pixel[c]) {
            this->med[col][c]++;
        }
    }
}

int PBAS::is_shadow(int col){
    //cout<< "rgb" << this->i_rgb[col] <<endl;
    Vec3b frame_rgb_pixel = i_rgb[col];
    Vec3b median_rgb_pixel = med[col];
    Mat med_pixel(1,1,CV_8UC3, &median_rgb_pixel);
    Mat rgb_pixel(1,1,CV_8UC3, &frame_rgb_pixel);
    Mat frame_hsv_pixel_mat;
    Mat median_hsv_pixel_mat;
    cvtColor(rgb_pixel, frame_hsv_pixel_mat, cv::COLOR_RGB2HSV);
    cvtColor(med_pixel, median_hsv_pixel_mat, cv::COLOR_RGB2HSV);
    //cout<< i_hsv.at<Vec3b>(0,0) <<endl;
    Vec3b median_hsv_pixel = median_hsv_pixel_mat.at<Vec3b>(0,0);
    Vec3b frame_hsv_pixel = frame_hsv_pixel_mat.at<Vec3b>(0,0);

    float h_m = median_hsv_pixel[0];
    float s_m = median_hsv_pixel[1];
    float v_m = median_hsv_pixel[2]==0?1:median_hsv_pixel[2];

    float h_f = frame_hsv_pixel[0];
    float s_f = frame_hsv_pixel[1];
    float v_f = frame_hsv_pixel[2];

    //cout<< "TAU_H " << abs(h_m-h_f) <<endl;
    //cout<< "TAU_S " << abs(s_m-s_f) <<endl;
    //cout<< "V " << v_f/v_m <<endl;
    
    if(abs(h_m-h_f)<TAU_H && abs(s_m-s_f)<TAU_S && ALPHA<=(float)v_f/v_m && (float)v_f/v_m<=BETA){
        return 1;
    }
    return 0;
}
