#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif //SERVER_APP_PROCESSING_H

#include "opencv2/opencv.hpp"
using namespace cv;

class ProcessingPipeline {
    Mat input_image;
public:
    explicit ProcessingPipeline(Mat input_image);
    Mat execute();
};

Mat page_frame_detection_filtering_based(const Mat& input_image, int LARGE_BLUR_KERNEL_SIZE = 51,
                                         int SMALL_BLUR_KERNEL_SIZE = 7, int BIN_BLUR_TH = 10, int EROSION_BLUR_TH = 30);
Mat page_frame_detection_stats_based(const Mat& input_image, int LARGE_BLUR_KERNEL_SIZE = 51,
                                     int SMALL_BLUR_KERNEL_SIZE = 7, int EROSION_BLUR_TH = 30);
