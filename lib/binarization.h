#ifndef SERVER_APP_BINARIZATION_H
#define SERVER_APP_BINARIZATION_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

class StatisticsBasedBinarization {
public:
    static int BLOCK_SIZE;
    static int CHUNK_SIZE;
    static int CORRECTION_OFFSET;

    explicit StatisticsBasedBinarization(int block_size, int chunk_size, int correction_offset);
    static Mat binarize_image(const Mat &input_image);
};

class FilteringBasedBinarization {
public:
    static int BLOCK_SIZE;
    static int CORRECTION_OFFSET;
    static int BLUR_KERNEL_SIZE;
    static int THRESHOLD;

    explicit FilteringBasedBinarization(int block_size, int correction_offset, int blur_kernel_size, int threshold,
                                        int hp_kernel_size);
    static Mat binarize_image(const Mat &input_image);
};