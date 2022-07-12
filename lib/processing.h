#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif //SERVER_APP_PROCESSING_H

#include "opencv2/opencv.hpp"
using namespace cv;

Rect erosion(const Mat &filtered_image, int EROSION_BOUNDARY = 150);
Mat stats_based_image_binarization(const Mat &input_image, int BLOCK_SIZE = 9, int CHUNK_SIZE = 37, int CORRECTION_OFFSET = 10);
Mat filtering_based_image_binarization(const Mat &input_image, const Mat &filtered_image, int BLOCK_SIZE = 19,  int CORRECTION_OFFSET = 10);
Mat edge_detection(const Mat& input_image, int KERNEL_SIZE = 11);