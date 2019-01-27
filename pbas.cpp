#include <iostream>
#include <math.h>
#include "pbas.h"

using namespace std;


PBAS::PBAS() {
    cout << "Ciao!" << endl;
}
PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200)
{
    cout << "Ciao!" << endl;
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

void PBAS::updateF(uint8_t *frameData, int x, int y, int stride) {
    Mat* B_copy;
    Mat* R_copy;
    
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
        *getPixelPtr(Fdata, x,y,Fstep) = 1;
    } else {
        *getPixelPtr(Fdata, x,y,Fstep) = 0;
        //updateB(frameData, x, y);
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
            B.push_back(b_elem);

            Mat d_elem = Mat::zeros(h, w, CV_8UC1);
            D.push_back(d_elem);

        }
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        R = Mat::zeros(h, w, CV_32FC1);
    }

    for(int x = 0; x < h; x++)
        for(int y = 0; y < w; y++)
        {
            updateF(frameData, x,y,stride);
            //updateR(frame, x,y);
            //updateT(frame, x,y);
        }
    
    return &F;
}

// int main(int argc, char const *argv[]){
//     PBAS* p = new PBAS();

// }
