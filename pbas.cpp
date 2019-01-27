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

Mat* PBAS::process(Mat* frame) {
    w = frame->cols;
    h = frame->rows;
    int stride = frame->step;
    // data stores pixel values and can be used for fast access by pointer
    uint8_t *frameData = frame->data;

    for(int x = 0; x < h; x++)
        for(int y = 0; y < w; y++)
        {
            updateF(frameData, x,y,stride);
            //updateR(frame, x,y);
            //updateT(frame, x,y);
        }
    
    return &F;
}
