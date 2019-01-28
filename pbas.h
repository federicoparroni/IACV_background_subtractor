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

        Mat frame;
        int w;
        int h;
        vector<Mat> B;
        Mat R;
        vector<Mat> D;
        Mat T;
        Mat F;
        Mat d_minavg;
        vector<pair<int,int>> displacement_vec;

        const uint8_t *i;
        uint8_t *q;
        float *r;
        float *t;

        void init();

        float distance(uint8_t, uint8_t);

        void updateF(int x, int y, int i_ptr);
        void updateB(int x, int y, int i_ptr);
        void updateR(int x, int y, int n, int i_ptr);
        void updateR_notoptimized(int x, int y, int i_ptr);
        void updateT(int x, int y, int i_ptr);
        void init_Mat(Mat matrix, float initial_value);
    public:
        PBAS();
        PBAS(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper);
        ~PBAS();

        Mat* process(const Mat frame);

};
