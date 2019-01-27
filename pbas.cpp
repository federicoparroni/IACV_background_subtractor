#include <iostream>
#include <math.h>
#include "pbas.h"
#include <stdlib.h>
#include <time.h>
#include <utility>       

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
    update_p = T.at<double>(x,y)*100;

    if(rand_numb > update_p){
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
        
        updateR(frame, x, y, n);
        updateR(frame, x+x_disp, y+y_disp, n);

    }

}

void PBAS::updateR(Mat* frame, int x, int y, int n){
    cout << "im in" << endl;
}

Mat* PBAS::process(Mat* frame) {
    w = frame->cols;
    h = frame->rows;
    int stride = frame->step;
    // data stores pixel values and can be used for fast access by pointer
    uint8_t *frameData = frame->data;

    // B, D, d_minavg T initialization
    if (B.size() == 0) {
        for(int i=0; i<N; i++) {
            Mat b_elem(h, w, CV_32FC1);
            randn(b_elem, Scalar(0.0), Scalar(1));
            B.push_back(b_elem);

            Mat d_elem = Mat::zeros(h, w, CV_32FC1);
            D.push_back(d_elem);

        }
        d_minavg = Mat::zeros(h, w, CV_32FC1);
        T = Mat::zeros(h, w, CV_32FC1);
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
