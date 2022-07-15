#ifndef SERVER_APP_PRE_PROCESSING_H
#define SERVER_APP_PRE_PROCESSING_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

class PreProcessing {

public:
    static int BLUR_KERNEL_SIZE;
    static int THRESHOLD;
    static int HP_KERNEL_SIZE;

    explicit PreProcessing(int blur_kernel_size, int threshold);
};

Mat pre_process_image(const Mat &input_image);
Mat edge_detection(const Mat &input_image);
