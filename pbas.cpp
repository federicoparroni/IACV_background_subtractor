#include <iostream>
#include <math.h>
#include "pbas.h"
#include <stdlib.h>
#include <time.h>
#include <utility>      
#include <algorithm> 

using namespace std;


PBAS::PBAS() {
    N = 30;
    K = 2;
    R_incdec = 0.05;
    R_lower = 18;
    R_scale = 5;
    T_dec = 0.05;
    T_inc = 1;
    T_lower = 2;
    T_upper = 200;
}
PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200)
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
}

PBAS::~PBAS() {
    B.clear();
    R.release();
    D.clear();
    T.release();
    F.release();
    d_minavg.release();
}

uint8_t PBAS::getPixel(uint8_t *data, int x, int y, int stride) {
    return data[x * stride + y];
}
uint8_t* PBAS::getPixelPtr(uint8_t *data, int x, int y, int stride) {
    return &data[x * stride + y];
    //return data + (x * stride + y) * sizeof(uchar);
}


float PBAS::distance(int a, int b) {
    return abs(a-b);
}

void PBAS::updateF(Mat* frame, int x, int y, int stride) {
    Mat* B_copy;
    Mat* R_copy;
    uint8_t* frameData = frame->data;
    
    uint8_t *Fdata = F.data;
    int Fstep = F.step;
    uint8_t *Rdata = R.data;
    int Rstep = R.step;

    //c = 0
    //while c < 3 or k >= self.K:
    int k = 0;       // number of lower-than-R distances for the channel 'c'
    int j = 0;
    while(j < N && k < K) {
        uint8_t *Bdata = B[j].data;
        int Bstep = B[j].step;
        //if(distance(frame[x,y], B_copy[j,x,y]) < R_copy[x,y]) {
        //cout << "Pixel: " << getPixel(frameData,x,y,stride) << endl;
        
        //if(distance(getPixel(frameData,x,y,stride), getPixel(B[j].data,x,y,Bstep)) < getPixel(Rdata,x,y,Rstep)) {
        if(distance(frame->at<double>(x,y), (int)B[j].at<uchar>(x,y)) < R.at<double>(x,y)) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y)
    if(k >= K) {
        //*getPixelPtr(Fdata, x,y,Fstep) = 1;
        F.at<uint8_t>(x,y) = 255;
    } else {
        //*getPixelPtr(Fdata, x,y,Fstep) = 0;
        F.at<uint8_t>(x,y) = 0;
        updateB(frame, x, y);
    }
}

void PBAS::updateR(Mat* frame, int x, int y, int n) {
    // find dmin
    int I = (int)frame->at<uchar>(x, y);
    int d_min = 255;
    int d_act = 0;
    for (int i=0; i<N; i++){
        d_act = distance(I, (int)B[i].at<uchar>(x, y));
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
    d_minavg.at<double>(x,y) = d_cum/double(N);
    cout << d_minavg.at<double>(x, y) << endl;

    // update R
    if (R.at<double>(x,y) > d_minavg.at<double>(x,y) * R_scale){
        R.at<double>(x,y) = R.at<double>(x,y)*(1 - R_incdec);
    } else {
        R.at<double>(x,y) = R.at<double>(x,y)*(1 + R_incdec);
    }
}

void PBAS::updateB(Mat* frame, int x, int y){
    int rand_numb, n, y_disp, x_disp;
    pair<int, int> disp;
    double update_p;
    vector<pair<int, int> > displacement_vec;
    
    displacement_vec.push_back(make_pair(-1, -1));
    displacement_vec.push_back(make_pair(-1, 1));
    displacement_vec.push_back(make_pair(-1, 0));
    displacement_vec.push_back(make_pair(1, -1));
    displacement_vec.push_back(make_pair(1, 1));
    displacement_vec.push_back(make_pair(1, 0));
    displacement_vec.push_back(make_pair(0, 1));
    displacement_vec.push_back(make_pair(0, -1));

    //initialize random seed
    srand (time(NULL));
    // generate a number between 0 and 99
    rand_numb = rand() %100;
    // get the T[x,y]
    update_p = (1/(T.at<double>(x,y)))*100;

    if(rand_numb < update_p){
        //generate a random number between 0 and N-1
        n = rand() % N;
        B[n].at<double>(x, y) = frame->at<double>(x, y);

        y_disp = 0;
        x_disp = 0;

        while((x_disp == 0 && y_disp == 0)||x+x_disp>=h||y+y_disp>=w){
            rand_numb = rand() %8;
            disp = displacement_vec[rand_numb];
            x_disp = disp.first;
            y_disp = disp.second;
        }

        B[n].at<double>(x+x_disp, y+y_disp) = frame->at<double>(x+x_disp, y+y_disp);
        
        //updateR(frame, x, y, n);
        //updateR(frame, x+x_disp, y+y_disp, n);

    }
}

void PBAS::updateT(int x, int y){
    float Tinc_over_dmin;
    Tinc_over_dmin = T_inc/d_minavg.at<double>(x,y);
    if(F.at<double>(x,y)==1)
        T.at<double>(x,y) += Tinc_over_dmin;
    else
        T.at<double>(x,y) -= Tinc_over_dmin;
    T.at<double>(x,y) = max((double)T_lower, T.at<double>(x,y));
    T.at<double>(x,y) = min((double)T_upper, T.at<double>(x,y));   
}

Mat* PBAS::process(Mat* frame) {
    w = frame->cols;
    h = frame->rows;
    int stride = frame->step;
    // data stores pixel values and can be used for fast access by pointer
    uint8_t *frameData = frame->data;

    // B, D, d_minavg initialization
    if (B.size() == 0) {
        for(int i=0; i<N; i++) {
            Mat b_elem(h, w, CV_8UC1);
            randu(b_elem, 0, 255);
            //cout << (int)b_elem.at<uchar>(0,0) << endl;
            B.push_back(b_elem);

            Mat d_elem = Mat::zeros(h, w, CV_8UC1);
            D.push_back(d_elem);
        }
        F = Mat::zeros(h, w, CV_8UC1);
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        T = Mat::ones(h, w, CV_32FC1);
        R = Mat::zeros(h, w, CV_32FC1);
    }

    for(int x = 0; x < h; x++)
        for(int y = 0; y < w; y++)
        {
            updateF(frame, x,y,stride);
            updateT(x, y);
            //cout << x << endl;
        }
    
    return &F;
}

// int main(int argc, char const *argv[]){
//     PBAS* p = new PBAS();

// }
