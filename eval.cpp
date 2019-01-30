#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include "pbas.cpp"

namespace fs = std::experimental::filesystem;
using namespace std;

struct Metrics {
    float re;
    float sp;
    float fpr;
    float fnr;
    float pwc;
    float f1;
    float pr;
};

Metrics avg_metrics(vector<Metrics> metrics_vector){
    float re_overall = 0;
    float sp_overall = 0;
    float fpr_overall = 0;
    float fnr_overall = 0;
    float pwc_overall = 0;
    float f1_overall = 0;
    float pr_overall = 0;
    int N = metrics_vector.size();
    for(Metrics m : metrics_vector){
        re_overall += m.re;
        sp_overall += m.sp;
        fpr_overall += m.fpr;
        fnr_overall += m.fnr;
        pwc_overall += m.pwc;
        pr_overall += m.pr;
        f1_overall += m.f1;
    }
    return {re_overall/N, sp_overall/N, fpr_overall/N, fnr_overall/N, pwc_overall/N, f1_overall/N, pr_overall/N};
}

Metrics eval(string path) {
    cout << "## evaluating video: " + path << endl;

    string input_bp = path + "/input";
    string gt_bp = path + "/groundtruth";
    string temporal_ROI = path + "/temporalROI.txt";
    
    int tp = 0;
    int tn = 0;
    int fn = 0;
    int fp = 0;

    vector<fs::path> inputs; // ordered array of input images
    copy(fs::directory_iterator(input_bp), fs::directory_iterator(), back_inserter(inputs));
    sort(inputs.begin(), inputs.end()); 
    vector<fs::path> gts; // ordered array of groundtruths
    copy(fs::directory_iterator(gt_bp), fs::directory_iterator(), back_inserter(gts));
    sort(gts.begin(), gts.end());

    ifstream infile(temporal_ROI); // read the temporal ROI
    int temporal_ROI_beg, temporal_ROI_end;
    infile >> temporal_ROI_beg >> temporal_ROI_end; 

    PBAS *pbas = new PBAS();

    for (int i=0; i < inputs.size(); i++){
        Mat input, gt, mask;
        input = imread(inputs[i].string(), CV_LOAD_IMAGE_COLOR);        break;

        cvtColor(input, input, cv::COLOR_RGB2GRAY);
        mask = *(pbas->process(input));

        if (i+1 >= temporal_ROI_beg && i+1 <= temporal_ROI_end) {
            gt = imread(gts[i].string(), CV_8UC1);
            tp += countNonZero(gt & mask);
            tn += countNonZero(~(gt | mask));
            fn += countNonZero(gt & (gt ^ mask));
            fp += countNonZero(mask & (gt ^ mask));
        }

        imshow("Frame", input);
        imshow("Mask", mask);
        char c=(char)waitKey(25);
        if(c==27) break;
    }
    destroyAllWindows();

    // relative to this video
    float re = (float)tp/(tp+fn);
    float pr = (float)tp/(tp+fp);
    float f1 = (2*pr*re)/(pr+re);
    float sp = (float)tn/(tn+fp);
    float fpr = (float)fp/(fp+tn);
    float fnr = (float)fn/(tn+fp);
    float pwc = 100*((float)fn+fp)/(tp+fn+fp+tn);

    // cout << "tp: " + to_string(tp) + " -- tn: " + to_string(tn) + " -- fn: " + to_string(fn) + " -- fp: " + to_string(fp) + "\n" << endl;
    // cout << "re: " + to_string(re) + " -- sp: " + to_string(sp) + " -- fpr: " + to_string(fpr) + " -- fnr: " + to_string(fnr) + " -- pwc: " + to_string(pwc) + " -- f1: " + to_string(f1) + " -- pr: " + to_string(pr) << endl;
    
    return {re, sp, fpr, fnr, pwc, f1, pr};
}

int main() {
    string changedetection_path = "./dataset/dataset";
    vector<Metrics> metrics_video;
    vector<Metrics> metrics_category;
    
    for(const auto & entry : fs::directory_iterator(changedetection_path)) {
        cout << "\n \n# evaluating category: " + entry.path().string() + "\n \n" << endl;
        for(const auto & vid : fs::directory_iterator(entry)) {
            metrics_video.push_back(eval(vid.path().string()));
        }
        Metrics m = avg_metrics(metrics_video);
        metrics_category.push_back(m);
        cout << "\n re: " + to_string(m.re) + " -- sp: " + to_string(m.sp) + " -- fpr: " + to_string(m.fpr) + " -- fnr: " + to_string(m.fnr) + " -- pwc: " + to_string(m.pwc) + " -- f1: " + to_string(m.f1) + " -- pr: " + to_string(m.pr) << endl;
 
    }
    Metrics m_o = avg_metrics(metrics_category);
    cout << "\n \n re: " + to_string(m_o.re) + " -- sp: " + to_string(m_o.sp) + " -- fpr: " + to_string(m_o.fpr) + " -- fnr: " + to_string(m_o.fnr) + " -- pwc: " + to_string(m_o.pwc) + " -- f1: " + to_string(m_o.f1) + " -- pr: " + to_string(m_o.pr) << endl;
}