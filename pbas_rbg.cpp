#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <utility>      
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <chrono>
using namespace std::chrono;

using namespace std;
using namespace cv;

class PBAS
{
    private:
        int frame_count = 0;

        // model params
        int N;
        int K;
        float R_incdec;
        float R_lower;
        float R_scale;
        float T_dec;
        float T_inc;
        float T_lower;
        float T_upper;
        float alpha;
        vector<float> I_m;
        
        // frame
        vector<Mat> frame;
        vector<Mat> frame_grad;
        int w;
        int h;

        // model support matrices and pointers
        vector<uint8_t*> i;
        vector<float*> i_grad;

        vector<vector<Mat>> B;
        vector<vector<Mat>> B_grad;

        vector<Mat> R; vector<float*> r;
        vector<Mat> T; vector<float*> t;
    
        vector<vector<Mat>> D;
        vector<Mat> d_minavg;

        vector<pair<int,int>> displacement_vec;
        
        //Vec3b *med;

        // output
        vector<Mat> F; vector<uint8_t*> q;
        //Mat final_mask;

        // methods
        void init(int channels);
        double distance(double p, double p_grad, double g, double g_grad, int c);
        //void updateMedian(int col);

        void updateF(int x, int y, int c);
        void updateB(int x, int y, int c);
        void updateR(int x, int y, int c, int n);
        void updateR_notoptimized(int x, int y, int c, int n);
        void updateT(int x, int y, int c);

        void init_Mat(Mat &matrix, float initial_value);
        Mat gradient_magnitude(const Mat &frame);

    public:
        PBAS();
        PBAS(int N, int K, float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_lower, int T_upper, int alpha);
        ~PBAS();
        
        bool verbose = true;

        // main
        Mat process(const Mat &frame);
        
        // utils
        void showCVMat(Mat matrix, bool normalize, string window_name);

        //Mat median;
};


PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200, int alpha=10)
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
    this->alpha = alpha;
    //this->I_m = 1.0;
    
    //initialize random seed
    //srand (time(NULL));

    // initialize displacement array
    displacement_vec.push_back(make_pair(-1, -1));
    displacement_vec.push_back(make_pair(-1, 1));
    displacement_vec.push_back(make_pair(-1, 0));
    displacement_vec.push_back(make_pair(1, -1));
    displacement_vec.push_back(make_pair(1, 1));
    displacement_vec.push_back(make_pair(1, 0));
    displacement_vec.push_back(make_pair(0, 1));
    displacement_vec.push_back(make_pair(0, -1));
}

PBAS::~PBAS() {
    frame.clear();
    frame_grad.clear();
    //median.release();
    B.clear();
    R.clear();
    D.clear();
    T.clear();
    F.clear();
    d_minavg.clear();
    displacement_vec.clear();
}


void PBAS::init_Mat(Mat &matrix, float initial_value){
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            matrix.at<float>(i,j) = initial_value;
        }
    }
}

Mat PBAS::process(const Mat &frame) {
    split(frame, this->frame);
    //this->frame = frame;
    this->w = frame.cols;
    this->h = frame.rows;
    int channels = frame.channels();

    // B, D, d_minavg initialization
    if (B.size() == 0) {
        init(channels);
    }
    
    // gradients computation
    split(gradient_magnitude(frame), this->frame_grad);

    // init B
    // if(frame_count < N) {
    //     vector<Mat> init;
    //     calculateFeatures(&init, &this->frame_rgb);
    //     B.push_back(init[0]);
    //     B_grad.push_back(gradient_magnitude(&this->frame_rgb));
    //     frame_count++;
    //     return &F;
    // }

    auto start = high_resolution_clock::now();
    for(int c=0; c < channels; c++) {
        for(int x=0; x < this->h; x++) {
            this->i[c] = this->frame[c].ptr<uint8_t>(x);
            this->i_grad[c] = frame_grad[c].ptr<float>(x);
            this->q[c] = F[c].ptr<uint8_t>(x);
            this->r[c] = R[c].ptr<float>(x);
            this->t[c] = T[c].ptr<float>(x);
            //this->med = median.ptr<Vec3b>(x);

            this->I_m[c] = mean(this->frame_grad[c]).val[0];

            for (int y=0; y < this->w; y++) {
                //cout << "x,y" << x << ";" << y << endl;
                updateF(x, y, c);
                updateT(x, y, c);
                // q[y] = i[y][0];
            }
            
        }
        // showCVMat(frame_grad[c], false, format("framegrad%d",c));
        // showCVMat(R[c], false, format("R%d",c));
        // showCVMat(F[c], false, format("F%d",c));
        // showCVMat(T[c], false, format("T%d",c));
        //medianBlur(F[c],F[c],3);
    }
    //showCVMat(B[0][0], false, format("B%d",0));

    if(verbose) {
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << duration.count() << "ms" << endl;
    }

    Mat res = Mat::zeros(h, w, CV_8UC1);
    for(int c = 0; c < channels; c++) {
        bitwise_or(res, F[c], res);
    }
    
    return res;
}

// initialize all support matrices
void PBAS::init(int channels) {
    for(int c=0; c<channels; c++) {
        vector<Mat> b, b_grad, d;
        for(int n=0; n<N; n++) {
            Mat b_elem(h, w, CV_8UC1);
            randu(b_elem, 0, 255);
            b.push_back(b_elem);
            //b.push_back(this->frame[c].clone());

            Mat b_grad_elem(h, w, CV_32FC1);
            randu(b_grad_elem, 0, 255);
            b_grad.push_back(b_grad_elem);
            //b_grad.push_back(frame_grad.clone());

            d.push_back(Mat::zeros(h, w, CV_32FC1));
        }
        B.push_back(b);
        B_grad.push_back(b_grad);
        D.push_back(d);

        F.push_back(Mat::zeros(h, w, CV_8UC1));
        R.push_back(Mat::zeros(h, w, CV_32FC1));
        T.push_back(Mat::zeros(h, w, CV_32FC1));

        d_minavg.push_back(Mat::zeros(h, w, CV_32FC1));

        init_Mat(T[c], T_lower);
        init_Mat(R[c], 128);

        i.push_back(0);
        i_grad.push_back(0);
        q.push_back(0);
        r.push_back(0);
        t.push_back(0);
        I_m.push_back(1);
    }
    assert(B.size() == channels);
    assert(B_grad.size() == channels);
    assert(D.size() == channels);
    assert(d_minavg.size() == channels);
    assert(F.size() == channels);
    assert(R.size() == channels);
    assert(T.size() == channels);
}

Mat PBAS::gradient_magnitude(const Mat &frame){
    Mat grad;
    int scale = 1;
    int ddepth = CV_32F;
    Mat input; frame.convertTo(input, CV_32F);
    
    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Gradient X
    Sobel(frame, grad_x, ddepth, 1, 0, 3, scale, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    // Gradient Y
    Sobel(frame, grad_y, ddepth, 0, 1, 3, scale, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    // Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}

double PBAS::distance(double p, double p_grad, double g, double g_grad, int c) {
    // TO-DO
    // return (this->alpha/this->I_m[c]) * abs(p_grad - g_grad) + abs(p - g);
    //cout << abs(p - g) << endl;
    return abs(p - g);
}

void PBAS::updateF(int x, int y, int c) {
    int k = 0;  // number of lower-than-R distances found so far
    int j = 0;
    while(j < N && k < K) {
        // TO-DO
        if(distance(i[c][y], i_grad[c][y], B[c][j].at<uint8_t>(x,y), B_grad[c][j].at<float>(x,y), c) < r[c][y]) {
            k++;
        }
        j++;
    }
    // check if at least K distances are less than R(x,y) => background pixel
    if(k >= K) {
        q[c][y] = 0;
        // q_shadow_hsv[i_ptr]=0;
        updateB(x, y, c);
    } else {
        // if(!is_shadow(i_ptr)) q_shadow_hsv[i_ptr] = 255;
        // else q_shadow_hsv[i_ptr] = 0;
        q[c][y] = 255;
    }
}

void PBAS::updateB(int x, int y, int c) {
    int y_disp, x_disp;
    pair<int, int> disp;

    // get the T[x,y]
    float update_p = 100 / t[c][y];
    
    // generate a number between 0 and 99
    int rand_numb = rand() %100;

    //generate a random number between 0 and N-1
    int n = rand() % N;
    if(rand_numb < update_p) {
        updateR(x, y, c, n);
        B[c][n].at<uint8_t>(x, y) = i[c][y];
        B_grad[c][n].at<float>(x, y) = i_grad[c][y];
    }
    
    return;
    //generate a random number between 0 and 99
    rand_numb = rand() %100;
    if(rand_numb < update_p) {
        y_disp = 0;
        x_disp = 0;

        while((x_disp == 0 && y_disp == 0) || x+x_disp >= h || y+y_disp >= w || x+x_disp < 0 || y+y_disp < 0){
            rand_numb = rand() % 8;
            disp = displacement_vec[rand_numb];
            x_disp = disp.first;
            y_disp = disp.second;
        }
        updateR_notoptimized(x+x_disp, y+y_disp, c, n);
        B[c][n].at<uint8_t>(x+x_disp, y+y_disp) = frame[c].at<uint8_t>(x+x_disp, y+y_disp);
        B_grad[c][n].at<float>(x+x_disp, y+y_disp) = frame_grad[c].at<float>(x+x_disp, y+y_disp);
    }
}

void PBAS::updateR(int x, int y, int c, int n) {
    // find dmin
    float d_min = 255;
    float d_act = 0;
    for (int j=0; j<N; j++) {
        d_act = distance(i[c][y], i_grad[c][y], B[c][j].at<uint8_t>(x, y), B_grad[c][j].at<float>(x, y), c);
        if (d_act < d_min)
            d_min = d_act;
    }
    // update Dk
    D[c][n].at<float>(x, y) = d_min;

    // find davg
    float d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += D[c][i].at<float>(x, y);
    }
    d_minavg[c].at<float>(x,y) = d_cum/N;

    // update R
    if (r[c][y] > d_minavg[c].at<float>(x,y) * R_scale){
        r[c][y] = r[c][y] * (1 - R_incdec);
    } else {
        r[c][y] = r[c][y] * (1 + R_incdec);
    }
    r[c][y] = max(R_lower, r[c][y]);
}

void PBAS::updateR_notoptimized(int x, int y, int c, int n) {
    // find dmin
    uint8_t I = frame[c].at<uint8_t>(x, y);
    float I_grad = frame_grad[c].at<float>(x,y);
    float d_min = 255;
    float d_act = 0;
    for (int i=0; i<N; i++) {
        d_act = distance(I, I_grad, B[c][i].at<uint8_t>(x, y), B_grad[c][i].at<float>(x, y), c);
        if (d_act < d_min)
            d_min = d_act;
    }

    // update Dk
    D[c][n].at<float>(x, y) = d_min;

    // find davg
    float d_cum = 0;
    for (int i=0; i<N; i++){
        d_cum += D[c][i].at<float>(x, y);
    }
    d_minavg[c].at<float>(x,y) = d_cum/N;

    // update R
    if (R[c].at<float>(x,y) > d_minavg[c].at<float>(x,y) * R_scale){
        R[c].at<float>(x,y) = R[c].at<float>(x,y)*(1 - R_incdec);
    } else {
        R[c].at<float>(x,y) = R[c].at<float>(x,y)*(1 + R_incdec);
    }
    R[c].at<float>(x,y) = max(R_lower, R[c].at<float>(x,y));
}

void PBAS::updateT(int x, int y, int c) {
    float oldT = t[c][y];
    if(q[c][y] == 255)
        t[c][y] += T_inc / (d_minavg[c].at<float>(x,y) + 0.1);
    else
        t[c][y] -= T_dec / (d_minavg[c].at<float>(x,y) + 0.1);

    t[c][y] = max(T_lower, t[c][y]);
    t[c][y] = min(T_upper, t[c][y]);
}

// void PBAS::updateMedian(int col){
//     Vec3b med_pixel = this->med[col];
//     Vec3b frame_pixel = this->i_rgb[col];

//     for(uint8_t c=0; c < 3; c++) {    
//         if(med_pixel[c] > frame_pixel[c]) {
//             this->med[col][c]--;
//         }
//         if(med_pixel[c] < frame_pixel[c]) {
//             this->med[col][c]++;
//         }
//     }
// }


void PBAS::showCVMat(Mat matrix, bool normalize, string window_name){
    Mat matrix_p;
    double max_matrix;
    
    //initialize the matrix to print with the value of the matrix passed as parameter
    matrix_p = matrix.clone();
    if(normalize){
        //convert the matrix into double
        matrix_p.convertTo(matrix_p, CV_64FC1);
        // retrieve the maximum of the matrix and put it into max_matrix
        minMaxLoc(matrix, NULL, &max_matrix);
        // bring all the values between 0 and 1
        matrix_p /= max_matrix;
        // bring all the values between 0 and 255
        matrix_p *= 255;
    }

    //convert the value of the matrix into INT values
    matrix_p.convertTo(matrix_p, CV_8UC1);
    //show the matrix
    imshow(window_name, matrix_p);
}

// void PBAS::calculateFeatures(vector<Mat> *feature, Mat *inputImage)
// {
//   if (!feature->empty())
//     feature->clear();

//   Mat mag[3], dir;
//   if (inputImage->channels() == 3)
//   {
//     std::vector<cv::Mat> rgbChannels(3);
//     split(*inputImage, rgbChannels);

//     for (int l = 0; l < 3; ++l)
//     {
//       Sobel(rgbChannels.at(l), sobelX, CV_32F, 1, 0, 3, 1, 0.0);
//       Sobel(rgbChannels.at(l), sobelY, CV_32F, 0, 1, 3, 1, 0.0);

//       // Compute the L2 norm and direction of the gradient
//       cartToPolar(sobelX, sobelY, mag[l], dir, true);
//       feature->push_back(mag[l]);
//       sobelX.release();
//       sobelY.release();
//     }

//     feature->push_back(rgbChannels.at(0));
//     feature->push_back(rgbChannels.at(1));
//     feature->push_back(rgbChannels.at(2));
//     rgbChannels.at(0).release();
//     rgbChannels.at(1).release();
//     rgbChannels.at(2).release();
//   }
//   else
//   {
//     Sobel(*inputImage, sobelX, CV_32F, 1, 0, 3, 1, 0.0);
//     Sobel(*inputImage, sobelY, CV_32F, 0, 1, 3, 1, 0.0);

//     // Compute the L2 norm and direction of the gradient
//     cartToPolar(sobelX, sobelY, mag[0], dir, true);
//     feature->push_back(mag[0]);

//     Mat temp;
//     inputImage->copyTo(temp);
//     feature->push_back(temp);
//     temp.release();
//   }

//   mag[0].release();
//   mag[1].release();
//   mag[2].release();
//   dir.release();
// }

// ================
// +++  SHADOWS

/*
int PBAS::is_shadow(int col){
    Vec3b frame_rgb_pixel = i_rgb[col];
    Vec3b median_rgb_pixel = med[col];
    Mat med_pixel(1,1,CV_8UC3, &median_rgb_pixel);
    Mat rgb_pixel(1,1,CV_8UC3, &frame_rgb_pixel);
    Mat frame_hsv_pixel_mat;
    Mat median_hsv_pixel_mat;
    cvtColor(rgb_pixel, frame_hsv_pixel_mat, COLOR_RGB2HSV);
    cvtColor(med_pixel, median_hsv_pixel_mat, COLOR_RGB2HSV);
    
    Vec3b median_hsv_pixel = median_hsv_pixel_mat.at<Vec3b>(0,0);
    Vec3b frame_hsv_pixel = frame_hsv_pixel_mat.at<Vec3b>(0,0);

    float h_m = median_hsv_pixel[0];
    float s_m = median_hsv_pixel[1];
    float v_m = median_hsv_pixel[2]==0?1:median_hsv_pixel[2];

    float h_f = frame_hsv_pixel[0];
    float s_f = frame_hsv_pixel[1];
    float v_f = frame_hsv_pixel[2];
    
    if(abs(h_m-h_f)<TAU_H && abs(s_m-s_f)<TAU_S && ALPHA<=(float)v_f/v_m && (float)v_f/v_m<=BETA){
        return 1;
    }
    return 0;
}

project a HLS pixel into the euclidean h,s,L space
Vec3d PBAS::tohsLprojection(Vec3b pixel) {
    Mat rgb_container(1,1,CV_8UC3, &pixel);
    Mat hsl_container;
    cvtColor(rgb_container, hsl_container, COLOR_RGB2HLS);
    Vec3b hsl_pixel = hsl_container.at<Vec3b>(0,0);

    Vec3d res;
    double H = hsl_pixel[0] * M_PI / 90;
    uint8_t L = hsl_pixel[1];
    uint8_t S = hsl_pixel[2];
    res[0] = S * cos(H);
    res[1] = S * sin(H);
    res[2] = L;
    return res;
}

double PBAS::hsLproduct(Vec3b p1, Vec3b p2) {
    int dot1 = p1[0]*p2[0];
    int dot2 = p1[1]*p2[1];
    return max(0,dot1+dot2) + p1[2]*p2[2];
}

void PBAS::color_normalized_cross_correlation() {
    int M = 7, N = 7; int MN = M*N;
    int halfM = (M-1)/2; int halfN = (N-1)/2;
    float cncc_threshold = 0.9;

    shadow_cncc = Mat::zeros(h, w, CV_8UC1);

    for(int x=halfM; x<this->h - halfM; ++x) {
    for(int y=halfN; y<this->w - halfN; ++y) {
        if(F.at<uint8_t>(x,y) != 0) {
            // Vec3b *f = frame_hsl.ptr<Vec3b>(0);
            // Vec3b *b = bg_hsl.ptr<Vec3b>(0);
            double cncc = 0;
            double L_avg_frame = 0; double L_avg_bg = 0;
            double var_f = 0; double var_bg = 0;
            for(int i=x-halfM; i<x+halfM; ++i) {
            for(int j=y-halfN; j<y+halfN; ++j) {
                Vec3d pixelF_ij = frame_hsl.at<Vec3d>(i,j);
                Vec3d pixelB_ij = bg_hsl.at<Vec3d>(i,j);
                cncc += hsLproduct(pixelF_ij, pixelB_ij);
                L_avg_frame += pixelF_ij[2];
                L_avg_bg += pixelB_ij[2];
                var_f += hsLproduct(pixelF_ij, pixelF_ij);
                var_bg += hsLproduct(pixelB_ij, pixelB_ij);
            }
            }
            L_avg_frame /= MN;
            L_avg_bg /= MN;
            var_f -= MN * L_avg_frame * L_avg_frame;
            var_bg -= MN * L_avg_bg * L_avg_bg;

            cncc -= MN * L_avg_frame * L_avg_bg;
            cncc /= sqrt(var_f * var_bg);

            if(cncc < cncc_threshold) {
                shadow_cncc.at<uint8_t>(x,y) = 255;
            }
        }
    }
    }
    return;
}
*/