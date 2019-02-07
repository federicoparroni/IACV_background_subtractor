#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/stat.h> 
#include "pbas.cpp"

using namespace std;
using namespace cv;

int state;
int starting_minute;
bool save_frames;
vector<tuple<int, int>> time_slots;
int capture_every;
string folder_name;

void go_to(int s){
    switch (s) {
        case 0:
            state = 0;
            cout << "want to save frames? (y/n)" << endl;
            break;
        case 1:
            state = 1;
            cout << "put starting minute in which to save (x) or enter to go ahead" << endl;
            break;
        case 2:
            state = 2;
            cout << "put ending minute in which stop to save (x)" << endl;
            break;
        case 3:
            state = 3;
            cout << "save a frame every? (x, in seconds). enter for default value 1" << endl;
            break;
        case 4:
            state = 4;
            cout << "select a folder name in which to save (enter for a default name)" << endl;
            break;
    }
}

int process(string & line){
    switch (state)
    {
        case 0:
            if(line == ""){
                go_to(0);
            }
            if(line == "y"){
                go_to(1);
                save_frames = true;
            }
            else{
                return -1;
            }
            break;
        case 1:
            if(line == ""){
                go_to(3);
            }
            else{
                starting_minute = stoi(line);
                go_to(2);
            }
            break;
        case 2:
            if(line == ""){
                go_to(2);
            }
            else{
                time_slots.push_back(make_tuple(starting_minute, stoi(line)));
                go_to(1);
            }
            break;
        case 3:
            if(line == ""){
                capture_every = 1;
                go_to(4);
            }
            else{
                capture_every = stoi(line);
                go_to(4);
            }
            break;
        case 4:
            folder_name = line;
            return -1;
            break;
    }
}

int main(int argc, char const *argv[]) {
    // init
    string filename = "videos/camera1.mp4";
    if(argc >= 2) filename = argv[1];
    int N = 30;
    int K = 3;
    float R_incdec = 0.05;
    int R_lower = 18;
    int R_scale = 5;
    float T_dec = 0.05;
    int T_inc = 1;
    int T_lower = 2;
    int T_upper = 200;
    int alpha = 10;
    if(argc == 12) {
        N = stoi(argv[2]);
        K = stoi(argv[3]);
        R_incdec = stof(argv[4]);
        R_lower = stoi(argv[5]);
        R_scale = stoi(argv[6]);
        T_dec = stof(argv[7]);
        T_inc = stoi(argv[8]);
        T_lower = stoi(argv[9]);
        T_upper = stoi(argv[10]);
        alpha = stoi(argv[11]);
    }
    VideoCapture cap(filename);
    PBAS *pbas = new PBAS(N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha);
    Mat frame;
    Mat* mask;
    Mat edges;
    float fps = cap.get(CV_CAP_PROP_FPS);
    float act_seconds = 0.0, partial_seconds = -1 / fps;
    int count_time_slot = 0;
    string folder_base, act_subfolder_base;
    
    // menu
    go_to(0);
    for (std::string line; std::cout << "in > " && std::getline(std::cin, line); )
    {
        if (process(line) == -1){
            break;
        }
    }

    // create dirs
    if(save_frames){
        mkdir("./frames", 0777);
        if(folder_name == ""){
            folder_base = "./frames/" + folder_base + filename.substr(filename.rfind('/') + 1, filename.rfind('.') - filename.rfind('/') - 1);
        }
        else{
            folder_base = "./frames/" + folder_name;
        }
        mkdir(folder_base.c_str(), 0777);
        act_subfolder_base = folder_base + "/" + to_string(get<0>(time_slots[count_time_slot])) + "_" + to_string(get<1>(time_slots[count_time_slot]));
        mkdir(act_subfolder_base.c_str(), 0777);
    }

    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        //GaussianBlur(frame, frame, Size(3,3), 0, 0);

        mask = pbas->process(&frame);
        imshow("Frame", frame);
        moveWindow("Frame", 20,20);

        imshow("Mask", *mask);

        // moveWindow("Mask", 372,20);
        moveWindow("Mask", 420, 20);

        // save frame
        if (save_frames) {
            act_seconds += 1 / fps;
            cout << act_seconds << endl;
            if (act_seconds >= get<0>(time_slots[count_time_slot]) && act_seconds <= get<1>(time_slots[count_time_slot])){
                partial_seconds += 1 / fps;
                if(partial_seconds >= capture_every){
                    cout << "capturing frame" << endl;
                    Mat to_save, grey_frame;
                    cvtColor(frame, grey_frame, cv::COLOR_RGB2GRAY);
                    hconcat(grey_frame, *mask, to_save);
                    imwrite(act_subfolder_base + "/" + to_string(act_seconds) + ".jpg", to_save);
                    partial_seconds = 0;
                }
            }
            else if (act_seconds > get<1>(time_slots[count_time_slot]) && count_time_slot < time_slots.size() - 1){
                count_time_slot++;
                act_subfolder_base = folder_base + "/" + to_string(get<0>(time_slots[count_time_slot])) + "_" + to_string(get<1>(time_slots[count_time_slot]));
                mkdir(act_subfolder_base.c_str(), 0777);
            }
        }


        // imshow("Shadows CNCC", pbas->shadow_cncc);
        // moveWindow("Shadows CNCC", 750,20);

        // imshow("Shadows CNCC", pbas->shadow_cncc);
        // moveWindow("Shadows CNCC", 750,20);

        // imshow("shadow_hsv", pbas->F_shadow_hsv);
        // moveWindow("Shadows hsv", 350,400);

        // Mat converted;
        // pbas->median.convertTo(converted, CV_16SC3);
        // frame.convertTo(frame, CV_16SC3);
        // Mat res = abs(frame - converted);
        // res.convertTo(res, CV_8UC3);
        // cvtColor(res, res, COLOR_RGB2GRAY);
        // threshold(res, res, 50, 255, THRESH_BINARY);
        // imshow("bg model", res);
        // moveWindow("bg model", 350,400);

        char c=(char)waitKey(25);
        if(c==27) break;
    }
    pbas->~PBAS();
    cap.release();
    destroyAllWindows();
    return 0;
}
