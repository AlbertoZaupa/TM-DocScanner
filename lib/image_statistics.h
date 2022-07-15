#ifndef SERVER_APP_IMAGE_STATISTICS_H
#define SERVER_APP_IMAGE_STATISTICS_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

void block_mean(const Mat &m, unsigned char **mean_matrix, int block_size);
void block_stats(const Mat &m, unsigned char **mean_matrix, float **var_matrix, int block_size);
