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

float PBAS::distance(int a, int b) {
    return abs(a-b);
}

void updateF(Mat frame, int x, int y) {
    //c = 0
    //while c < 3 or k >= self.K:
    int k = 0;       // number of lower-than-R distances for the channel 'c'
    int j = 0;
    while(j < N || k >= K) {
        if(_distance(frame[x,y], B_copy[j,x,y]) < R_copy[x,y]) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y)
    if(k >= K) {
        F[x,y] = 1;
    } else {
        F[x, y] = 0;
        updateBg(frame, x, y);
    }
}

Mat* process(Mat* frame) {
    w = frame->size().width();
    h = frame->size().height();

    for(int x = 0; x < w; x++)
        for(int y = 0; y < h; y++)
        {
            updateF(frame, x,y);
            updateR(frame, x,y);
            updateT(frame, x,y);
        }
    
    return F;
}


int main() {
    PBAS *p = new PBAS();
    
    return 0;
}
