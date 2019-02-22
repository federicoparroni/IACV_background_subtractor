#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PBASgray
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
        Mat d_minavg;
        Mat final_mask;
        vector<pair<int,int>> displacement_vec;

        Mat frame_hsl; Vec3d *f_hsl_ptr;
        Mat bg_hsl; Vec3d *bg_hsl_ptr;

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

        // shadows with CNCC
        void color_normalized_cross_correlation();
        Vec3d tohsLprojection(Vec3b pixel);
        double hsLproduct(Vec3b p1, Vec3b p2);

        // shadows with paperino
        Mat shadows_corner(Mat* frame, Mat* mask);
    public:
        PBASgray();
        PBASgray(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper, int alpha);
        ~PBASgray();

        Mat* process(const Mat* frame);
        void showCVMat(Mat matrix, bool normalize, string window_name);

        Mat frame;
        Mat median;

        Mat shadow_cncc;
        Mat F_shadow_hsv;
        Mat shadow_corner;
};
