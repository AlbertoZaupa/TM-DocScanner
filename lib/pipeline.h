#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

class ProcessingPipeline {
    int LARGE_BLUR_KERNEL_SIZE;
    int SMALL_BLUR_KERNEL_SIZE;
    int BIN_BLUR_TH;
    int EROSION_BLUR_TH;
    int STATS_BLOCK_SIZE;
    int STATS_CHUNK_SIZE;
    int FILTERING_BLOCK_SIZE;
    int EDGE_DET_KERNEL_LENGTH;
    int CORRECTION_OFFSET;
    int EROSION_BOUNDARY;
public:
    explicit ProcessingPipeline(int large_blur_kernel_size = 51, int small_blur_kernel_size = 5, int bin_blur_th = 10,
                                int erosion_blur_th = 30, int stats_block_size = 9, int stats_chunk_size = 37,
                                int filtering_block_size = 19, int edge_det_kernel_length = 11,
                                int correction_offset = 10, int erosion_boundary = 200);
    Mat filtering_approach(const Mat& input_image);
    Mat statistical_approach(const Mat& input_image);
};
