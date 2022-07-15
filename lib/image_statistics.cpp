#include "image_statistics.h"
#include "opencv2/opencv.hpp"
using namespace cv;

void block_mean(const Mat &m, unsigned char **mean_matrix, int block_size) {
    if (!block_size%2) {
        std::cerr<<"image_statistics.block_mean(): The value of the block size must be an odd number\n";
        exit(1);
    }
    int offset = block_size/2;
    int block_area = block_size*block_size;

    long int block_rows_sum[m.size[1]];
    for (int i=0; i<m.size[1]; ++i) {
        block_rows_sum[i] = 0;
    }
    int gray_value;
    for (int col=0; col<m.size[1]; col++) {
        for (int row=0; row<block_size; ++row) {
            gray_value = m.at<unsigned char>(row, col);
            block_rows_sum[col] += gray_value;
        }
    }

    long int moving_sum; float mean;
    for (int i=offset; i<m.size[0]-offset; ++i) {
        if (i!=offset) {
            for (int j=0; j<m.size[1]; ++j) {
                gray_value = m.at<unsigned char>(i-offset-1, j);
                block_rows_sum[j] -= gray_value;
                gray_value = m.at<unsigned char>(i+offset, j);
                block_rows_sum[j] += gray_value;
            }
        }

        for (int j=offset; j<m.size[1]-offset; ++j) {
            if (j==offset) {
                moving_sum = 0;
                for (int k=0; k<block_size; ++k) {
                    moving_sum += block_rows_sum[k];
                }
            }
            else {
                moving_sum -= block_rows_sum[j-offset-1];
                moving_sum += block_rows_sum[j+offset];
            }
            mean = (float) moving_sum / (float) block_area;
            mean_matrix[i][j] = (unsigned char) mean;
        }
    }
}

void block_stats(const Mat &m, unsigned char **mean_matrix, float **var_matrix, int block_size) {
    if (!block_size%2) {
        std::cerr<<"image_statistics.block_stats(): The value of the block size must be an odd number\n";
        exit(1);
    }
    int offset = block_size/2;
    int block_area = block_size*block_size;

    long int block_rows_sum[m.size[1]];
    long int block_rows_squares_sum[m.size[1]];
    for (int i=0; i<m.size[1]; ++i) {
        block_rows_sum[i] = 0;
        block_rows_squares_sum[i] = 0;
    }
    int grey_value;
    for (int col=0; col<m.size[1]; col++) {
        for (int row=0; row<block_size; ++row) {
            grey_value = m.at<unsigned char>(row, col);
            block_rows_sum[col] += grey_value;
            block_rows_squares_sum[col] += grey_value*grey_value;
        }
    }

    long int moving_sum, moving_squares_sum; float mean;
    for (int i=offset; i<m.size[0]-offset; ++i) {
        if (i!=offset) {
            for (int j=0; j<m.size[1]; ++j) {
                grey_value = m.at<unsigned char>(i-offset-1, j);
                block_rows_sum[j] -= grey_value;
                block_rows_squares_sum[j] -= grey_value*grey_value;
                grey_value = m.at<unsigned char>(i+offset, j);
                block_rows_sum[j] += grey_value;
                block_rows_squares_sum[j] += grey_value*grey_value;
            }
        }

        for (int j=offset; j<m.size[1]-offset; ++j) {
            if (j==offset) {
                moving_sum = 0;
                moving_squares_sum = 0;
                for (int k=0; k<block_size; ++k) {
                    moving_sum += block_rows_sum[k];
                    moving_squares_sum += block_rows_squares_sum[k];
                }
            }
            else {
                moving_sum -= block_rows_sum[j-offset-1];
                moving_sum += block_rows_sum[j+offset];
                moving_squares_sum -= block_rows_squares_sum[j-offset-1];
                moving_squares_sum += block_rows_squares_sum[j+offset];
            }
            mean = (float) moving_sum / (float) block_area;
            mean_matrix[i][j] = (unsigned char) mean;
            var_matrix[i][j] = ((float) moving_squares_sum / (float) block_area) - mean*mean;
        }
    }
}