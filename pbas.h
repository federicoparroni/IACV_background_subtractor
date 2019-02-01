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
        int alpha;
        float I_m;

        float TAU_H;
        float TAU_S;
        float ALPHA;
        float BETA;

        Mat median;
        Mat frame;
        Mat frame_rgb;
        Mat frame_grad;
        int w;
        int h;
        vector<Mat> B;
        vector<Mat> B_grad;
        Mat R;
        vector<Mat> D;
        Mat T;
        Mat F;
        Mat F_shadow_hsv;
        Mat d_minavg;
        Mat final_mask;
        vector<pair<int,int>> displacement_vec;

        const uint8_t *i;
        const Vec3b *i_rgb;
        const uint8_t *i_grad;
        uint8_t *q;
        uint8_t *q_shadow_hsv;
        float *r;
        float *t;
        Vec3b *med;

        void init();

        float distance(uint8_t, uint8_t);
        float distance(uint8_t p, uint8_t p_grad, uint8_t g, uint8_t g_grad);

        void updateMedian(int col);
        void updateF(int x, int y, int i_ptr);
        void updateB(int x, int y, int i_ptr);
        void updateR(int x, int y, int n, int i_ptr);
        void updateR_notoptimized(int x, int y, int i_ptr);
        void updateT(int x, int y, int i_ptr);
        void init_Mat(Mat* matrix, float initial_value);
        Mat gradient_magnitude(Mat* frame);
        int is_shadow(int col);
        Mat shadows_corner(Mat* frame, Mat* mask);
    public:
        PBAS();
        PBAS(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper, int alpha);
        ~PBAS();

        Mat* process(const Mat* frame);

};
