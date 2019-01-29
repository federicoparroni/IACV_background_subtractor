#include <string>
#include <iostream>
#include <experimental/filesystem>
#include "pbas.cpp"

namespace fs = std::experimental::filesystem;
using namespace std;

void eval(string path){
    cout << "## evaluating video: " + path << endl;

    string input_bp = path + "/input";
    string gt_bp = path + "/groundtruth";
    
    vector<fs::path> inputs; // ordered array of input images
    copy(fs::directory_iterator(input_bp), fs::directory_iterator(), back_inserter(inputs));
    sort(inputs.begin(), inputs.end()); 
    vector<fs::path> gts; // ordered array of groundtruths
    copy(fs::directory_iterator(gt_bp), fs::directory_iterator(), back_inserter(gts));
    sort(gts.begin(), gts.end());

    PBAS *pbas = new PBAS();

    for (int i = 0; i < inputs.size(); i++){
        Mat input, gt, mask;
        input = imread(inputs[i].string(), CV_LOAD_IMAGE_COLOR);
        cvtColor(input, input, cv::COLOR_RGB2GRAY);
        gt = imread(gts[i].string(), CV_8UC1);

        mask = *(pbas->process(input));

        imshow("Frame", input);
        imshow("Mask", mask);
        char c=(char)waitKey(25);
        if(c==27) break;
    }
    destroyAllWindows();

}

int main()
{
    string changedetection_path = "./dataset/dataset";
    for (const auto & entry : fs::directory_iterator(changedetection_path)) {
        cout << "\n \n# evaluating category: " + entry.path().string() + "\n \n" << endl;
        for (const auto & vid : fs::directory_iterator(entry)) {
            eval(vid.path().string());
        }
    }

}