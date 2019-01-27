#include <iostream>
#include <math.h>
#include "pbas.h"

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

PBAS::~PBAS() {}

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
    while(j < N || k >= K) {
        uint8_t *Bdata = B[j].data;
        int Bstep = B[j].step;
        //if(distance(frame[x,y], B_copy[j,x,y]) < R_copy[x,y]) {
        if(distance(getPixel(frameData,x,y,stride), getPixel(B[j].data,x,y,Bstep)) < getPixel(Rdata,x,y,Rstep)) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y)
    if(k >= K) {
        //*getPixelPtr(Fdata, x,y,Fstep) = 1;
        F.at<uint8_t>(Point(x,y)) = 1;
    } else {
        //*getPixelPtr(Fdata, x,y,Fstep) = 0;
        F.at<uint8_t>(Point(x,y)) = 0;
        updateB(frame, x, y);
    }
}

void PBAS::updateR(Mat* frame, int x, int y, int n) {
    cout << "im in" << endl;
}

Mat* PBAS::process(Mat* frame) {
    w = frame->cols;
    h = frame->rows;
    int stride = frame->step;
    // data stores pixel values and can be used for fast access by pointer
    uint8_t *frameData = frame->data;

    // B, D, d_minavg initialization
    if (!B.size()) {
        for(int i=0; i<N; i++) {
            Mat b_elem(h, w, CV_32FC1);
            randn(b_elem, Scalar(0.0), Scalar(1));
            B.push_back(b_elem);

            Mat d_elem = Mat::zeros(h, w, CV_32FC1);
            D.push_back(d_elem);
        }
        F = Mat::zeros(h, w, CV_8UC1);
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        R = Mat::zeros(h, w, CV_32FC1);
    }

    for(int x = 0; x < h; x++)
        for(int y = 0; y < w; y++)
        {
            updateF(frame, x,y,stride);
            //updateR(frame, x,y);
            //updateT(frame, x,y);
        }
    
    return &F;
}

// int main(int argc, char const *argv[]){
//     PBAS* p = new PBAS();

// }
