#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

Rect erosion(const Mat &filtered_image, int EROSION_BOUNDARY);
Mat stats_based_image_binarization(const Mat &input_image, int BLOCK_SIZE, int CHUNK_SIZE, int CORRECTION_OFFSET);
Mat filtering_based_image_binarization(const Mat &input_image, const Mat &filtered_image, int BLOCK_SIZE,  int CORRECTION_OFFSET);
Mat edge_detection(const Mat& input_image, int KERNEL_SIZE);