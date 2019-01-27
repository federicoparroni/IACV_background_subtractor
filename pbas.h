#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PBAS
{
    private:
        int N;
        int K;
        float R_incdec;
        int R_lower;
        int R_scale;
        float T_dec;
        int T_inc;
        int T_lower;
        int T_upper;

        int w;
        int h;
        vector<Mat> B;
        Mat R;
        vector<Mat> D;
        Mat T;
        Mat F;
        Mat d_minavg;

        uint8_t getPixel(uint8_t *data, int x, int y, int stride);
        uint8_t* getPixelPtr(uint8_t *data, int x, int y, int stride);
        float distance(int, int);

        void updateF(Mat* frame, int x, int y, int stride);
        void updateB(Mat* frame, int x, int y);
        void updateR(Mat* frame, int x, int y, int n);
        void updateT(int x, int y);
        void init_Mat(Mat matrix, float initial_value);
    public:
        PBAS();
        PBAS(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper);
        ~PBAS();

        Mat* process(Mat* frame);

};
