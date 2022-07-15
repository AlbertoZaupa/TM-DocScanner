#include "utility.h"
#include "iostream"
#include "opencv2/opencv.hpp"

using namespace cv;

/*
 These functions print the values of a matrix to the screen
*/

void print_matrix(unsigned char** mat, int y_low, int y_high, int x_low, int x_high) {
    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            std::cout<<(int) mat[i][j]<<" ";
        }
        std::cout<<"\n";
    }
}

void print_matrix(float** mat, int y_low, int y_high, int x_low, int x_high) {
    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            std::cout<<mat[i][j]<<" ";
        }
        std::cout<<"\n";
    }
}

/*
 This function computes the median value of a matrix
*/

int mmedian(unsigned char** mat, int y_low, int y_high, int x_low, int x_high) {
    bool occurrences_array[256]; for (int i=0; i<256; ++i) occurrences_array[i] = false;
    int occurring_values = 0;
    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            if (!occurrences_array[mat[i][j]]) {
                occurrences_array[mat[i][j]] = true;
                occurring_values++;
            }
        }
    }

    int occurred_values[occurring_values];
    int next_index = 0;
    for (int i=0; i<256; ++i) {
        if (occurrences_array[i]) {
            occurred_values[next_index++] = i;
        }
    }

    return occurred_values[occurring_values/2 + 1*(occurring_values%2)];
}

/*
 This function computes the median value of an image
*/

int imedian(Mat m) {
    bool occurrences_array[256]; for (int i=0; i<256; ++i) occurrences_array[i] = false;
    int occurring_values = 0;
    m.forEach<unsigned char>([&occurrences_array, &occurring_values] (unsigned char &value, const int* p) -> void {
        if (!occurrences_array[value]) {
            occurrences_array[value] = true;
            occurring_values++;
        }
    });

    int occurred_values[occurring_values];
    int next_index = 0;
    for (int i=0; i<256; ++i) {
        if (occurrences_array[i]) {
            occurred_values[next_index++] = i;
        }
    }
    return occurred_values[occurring_values/2 + 1*(occurring_values%2)];
}

/*
 The function computes the mode of a matrix
*/

int mmode(unsigned char** mat, int y_low, int y_high, int x_low, int x_high) {
    int histogram[256]; for (int i=0; i<256; ++i) histogram[i] = 0;

    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            histogram[mat[i][j]]++;
        }
    }

    int mode = 0;
    for (int i=0; i<256; ++i) {
        if (histogram[i] > histogram[mode]) {
            mode = i;
        }
    }

    return mode;
}

/*
 The function computes the mode of an image
*/

int imode(Mat mat) {
    int histogram[256]; for (int i=0; i<256; ++i) histogram[i] = 0;

    for (int i=0; i<mat.size[0]; ++i) {
        for (int j=0; j<mat.size[1]; ++j) {
            histogram[mat.at<unsigned char>(i, j)]++;
        }
    }

    int mode = 0;
    for (int i=0; i<256; ++i) {
        if (histogram[i] > histogram[mode]) {
            mode = i;
        }
    }

    return mode;
}

/*
 The function computes the minimum value of an image
*/

unsigned char imin(Mat m) {
    int min = 255;
    int grey_value;
    for (int i=0; i<m.size[0]; i++) {
        for (int j=0; j<m.size[1]; ++j) {
            grey_value = m.at<unsigned char>(i, j);
            if (grey_value < min) min = grey_value;
        }
    }
    return min;
}

/*
 These functions computes the minimum of the two values
*/

int min(int a, int b) {
    return a > b ? b : a;
}

float min(float a, float b) {
    return a > b ? b : a;
}

/*
 These functions compute the maximum of the two values
*/

int max(int a, int b) {
    return a > b ? a : b;
}

float max(float a, float b) {
    return a > b ? a : b;
}

/*
 This function computes the mean value of a matrix
*/

float mmean(float **m, int y_low, int y_high, int x_low, int x_high) {
    float partial_mean, mean = 0;
    for (int i=y_low; i<y_high; ++i) {
        partial_mean = 0;
        for (int j=x_low; j<x_high; ++j) {
            partial_mean += m[i][j];
        }
        mean += partial_mean /= (x_high - x_low);
    }
    return mean / (y_high - y_low);
}

/*
 The function computes the local mean value of an image
*/

float mmean(Mat m, int y_low, int y_high, int x_low, int x_high) {
    long int partial_mean, mean = 0;
    for (int i=y_low; i<y_high; ++i) {
        partial_mean = 0;
        for (int j=x_low; j<x_high; ++j) {
            partial_mean += m.at<unsigned char>(i, j);
        }
        mean += partial_mean /= (x_high - x_low);
    }
    return mean / (y_high - y_low);
}

/*
 This function computes the mean of an image
*/

float imean(Mat mat) {
    long long int sum = 0;
    for (int i=0; i<mat.size[0]; ++i) {
        for (int j=0; j<mat.size[1]; ++j) {
            sum += mat.at<unsigned char>(i, j);
        }
    }
    return (float) sum / (float) (mat.size[0] * mat.size[1]);
}

/*
 This function computes the convolution of an image with a custom kernel
 */

void convolution(Mat m, float **dst, float **kernel, int k_height, int k_width) {
    if (!k_height%2 || !k_width%2) {
        std::cerr<<"utility.convolution(): The kernel's dimensions must be odd numbers\n";
        exit(1);
    }
    int y_offset = k_height/2;
    int x_offset = k_width/2;

    float moving_sum;
    for (int i=y_offset; i<m.size[0]-y_offset; ++i) {
        for (int j=x_offset; j<m.size[1]-x_offset; ++j) {
            moving_sum = 0;
            for (int k=0; k<k_height; ++k) {
                for (int v=0; v<k_width; ++v) {
                    moving_sum += kernel[k][v] * m.at<unsigned char>(i-y_offset+k, j-x_offset+v);
                }
            }
            dst[i][j] = moving_sum;
        }
    }
}

/*
 The function computes the standard deviation of a matrix
*/

float mstddev(unsigned char **m, unsigned char mean, int y_low, int y_high, int x_low, int x_high) {
    float partial_stddev, stddev = 0;
    for (int i=y_low; i<y_high; ++i) {
        partial_stddev = 0;
        for (int j=x_low; j<x_high; ++j) {
            partial_stddev += m[i][j] - mean;
        }
        stddev += partial_stddev / (x_high - x_low);
    }
    return stddev / (y_high - y_low);
}

/*
 The function computes the standard deviation of a portion of an image
*/

float mstddev(Mat m, unsigned char mean, int y_low, int y_high, int x_low, int x_high) {
    float partial_stddev, stddev = 0;
    for (int i=y_low; i<y_high; ++i) {
        partial_stddev = 0;
        for (int j=x_low; j<x_high; ++j) {
            partial_stddev += abs(m.at<unsigned char>(i, j) - mean);
        }
        stddev += partial_stddev / (x_high - x_low);
    }
    return stddev / (y_high - y_low);
}

/*
 The function computes the variance of a portion of an image
*/

float mvar(Mat m, unsigned char mean, int y_low, int y_high, int x_low, int x_high) {
    float partial_var, var = 0; int error;
    for (int i=y_low; i<y_high; ++i) {
        partial_var = 0;
        for (int j=x_low; j<x_high; ++j) {
            error = m.at<unsigned char>(i, j) - mean;
            partial_var += error*error;
        }
        var += partial_var / (x_high - x_low);
    }
    return var / (y_high - y_low);
}

/*
 The function computes, for each pixel of the image, the minimum value of a block centered on that pixel
*/

void block_min(Mat m, unsigned char **min_matrix, int block_size) {
    if (!block_size%2) {
        std::cerr<<"utility.block_min(): The value of the block size must be an odd number\n";
        exit(1);
    }
    int offset = block_size/2;
    
    Rect slice;
    for (int i=offset; i<m.size[0]-offset; i++) {
        for (int j=offset; j<m.size[1]-offset; ++j) {
            slice = Rect(j-offset, i-offset, block_size, block_size);
            min_matrix[i][j] = imin(m(slice));
        }
    }
}

/*
 The function computes the maximum value of a matrix
*/

float mmax(float **m, int y_low, int y_high, int x_low, int x_high) {
    float max = 0;
    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            if (m[i][j] > max) max = m[i][j];
        }
    }
    return max;
}

/*
 The function computes the local histogram of an image
*/

int* mhistogram(Mat m, int y_low, int y_high, int x_low, int x_high) {
    auto histogram = new int[256]; 
    for (int k=0; k<256; ++k) histogram[k] = 0;
    for (int i=y_low; i<y_high; ++i) {
        for (int j=x_low; j<x_high; ++j) {
            ++histogram[m.at<unsigned char>(i, j)];
        }
    }
    return histogram;
}

/*
 The function coumputes the local Othsu threshold of an image
*/

float othsu_threshold(int* histogram, int block_area) {
    float otsu_th = 0, var, max_var = 0, w1 = 0, w2 = 0, mu1 = 0, mu2 = 0;

    // The threshold value that maximizes between-class variance is the local otsu threshold
    for (int th=0; th<256; ++th) {
        for (int k=0; k<256; ++k) {
            if (k<=th) {
                w1 += histogram[k];
                mu1 += k*(double)histogram[k]/block_area;
            }
            else {
                w2 += histogram[k];
                mu2 += k*(double)histogram[k]/block_area;
            }
        }
        w1 /= block_area; w2 /= block_area;
        var = w1*w2*(mu1-mu2)*(mu1-mu2);
        if (var > max_var) {
            max_var = var; otsu_th = th;
        }
    }

    return otsu_th;
}

/*
 These functions rescale a matrix
 */

void rescale_matrix(const Mat& m, float prev_max, float desired_max) {
    m.forEach<uchar>([prev_max, desired_max] (uchar &value, const int* p) -> void {
        value = abs( value / prev_max) * desired_max;
    });
}

void rescale_matrix(const Mat& m, float desired_max) {
    double min_value, max_value;
    minMaxLoc(m, &min_value, &max_value);
    float prev_max = max(abs(min_value), abs(max_value));
    rescale_matrix(m, prev_max, desired_max);
}

/*
 This function calculates the histogram of a matrix and writes it on a file
*/

void histogram_to_file(const Mat& m, const char* path) {
    // The file is opened
    std::cout<<path<<"\n";
    FILE* file = fopen(path, "w");
    if (!file) {
        std::cerr<<"utility.histogram_to_file(): error opening the file\n";
        perror("Error: ");
        exit(1);
    }

    // The histogram of the image is calculated
    Mat histogram;
    int histogram_size = 256; float values_range[] = {0, 256};
    const float* hist_range[] = {values_range};
    calcHist(&m, 1, 0, Mat(), histogram, 1, &histogram_size, hist_range);

    // The histogram is written onto the file
    for (int i=0; i<256; ++i) {
        fprintf(file, "%f\n", histogram.at<float>(i));
        if (ferror(file)) {
            std::cerr<<"utility.histogram_to_file(): error writing on file\n";
            perror("Error: ");
            exit(1);
        }
    }
    fclose(file);
}
