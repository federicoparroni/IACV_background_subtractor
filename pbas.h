#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

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
        vector<cv::Mat> B;
        vector<int> R;
        vector<cv::Mat> D;
        vector<int> T;
        vector<int> F;
        float d_minavg;

    public:
        PBAS();
        PBAS(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper);
        ~PBAS();
};
