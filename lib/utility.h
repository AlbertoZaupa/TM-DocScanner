#ifndef SERVER_APP_UTILITY_H
#define SERVER_APP_UTILITY_H

#endif //SERVER_APP_UTILITY_H

#include "opencv2/opencv.hpp"
using namespace cv;

void print_matrix(unsigned char** mat, int y_low, int y_high, int x_low, int x_high);
void print_matrix(float** mat, int y_low, int y_high, int x_low, int x_high);
int mmedian(unsigned char** mat, int y_low, int y_high, int x_low, int x_high);
int imedian(Mat m);
int mmode(unsigned char** mat, int y_low, int y_high, int x_low, int x_high);
int min(int a, int b);
float min(float a, float b);
unsigned char imin(Mat m);
int max(int a, int b);
float max(float a, float b);
float mmean(float **m, int y_low, int y_high, int x_low, int x_high);
float mmean(Mat m, int y_low, int y_high, int x_low, int x_high);
void convolution(Mat m, float **dst, float **kernel, int k_height, int k_width);
float mstddev(float **m, float mean, int y_low, int y_high, int x_low, int x_high);
float mstddev(Mat m, float mean, int y_low, int y_high, int x_low, int x_high);
float mvar(Mat m, float mean, int y_low, int y_high, int x_low, int x_high);
void block_min(Mat m, unsigned char **min_matrix, int block_size);
void block_mean(Mat m, unsigned char **mean_matrix, int block_size);
void block_stats(Mat m, unsigned char **mean_matrix, float **var_matrix, int block_size);
float mmax(float** m, int y_low, int y_high, int x_low, int x_high);
int* mhistogram(Mat m, int y_low, int y_high, int x_low, int x_high);
float othsu_threshold(int* histogram, int block_area);
void rescale_matrix(const Mat& m, float desired_max);
void rescale_matrix(const Mat& m, float prev_max, float desired_max);
