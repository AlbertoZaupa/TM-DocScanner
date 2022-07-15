#include "pre_processing.h"
#include "opencv2/opencv.hpp"

int PreProcessing::BLUR_KERNEL_SIZE = 51;
int PreProcessing::THRESHOLD = 30;
int PreProcessing::HP_KERNEL_SIZE = 11;

Mat pre_process_image(const Mat &input_image) {

    Mat output_image = input_image.clone();

    medianBlur(output_image, output_image, PreProcessing::BLUR_KERNEL_SIZE);
    output_image = edge_detection(output_image);

    return output_image;
}

Mat edge_detection(const Mat &input_image) {
    // The filters' kernels are initialized
    Mat right_left_filter = Mat(1, PreProcessing::HP_KERNEL_SIZE, CV_32S);
    Mat left_right_filter = Mat(1, PreProcessing::HP_KERNEL_SIZE, CV_32S);
    Mat top_bottom_filter = Mat(PreProcessing::HP_KERNEL_SIZE, 1, CV_32S);
    Mat bottom_top_filter = Mat(PreProcessing::HP_KERNEL_SIZE, 1, CV_32S);
    //float filter_coefficient = (float) 2 / (float) KERNEL_SIZE; // each side of the filter calculates the mean of its side
    left_right_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[1] < PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else if (p[1] > PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    right_left_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[1] < PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else if (p[1] > PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });
    top_bottom_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[0] < PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else if (p[0] > PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    bottom_top_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[0] < PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else if (p[0] > PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });

    // The matrices where the output of the filtering process will reside
    Mat right_left_output = Mat::zeros(input_image.size(), CV_32F);
    Mat left_right_output = Mat::zeros(input_image.size(), CV_32F);
    Mat top_bottom_output = Mat::zeros(input_image.size(), CV_32F);
    Mat bottom_top_output = Mat::zeros(input_image.size(), CV_32F);
    Mat filtered_image = Mat::zeros(input_image.size(), 0);

    // All 4 filters are applied
    filter2D(input_image, left_right_output, -1, left_right_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, right_left_output, -1, right_left_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, top_bottom_output, -1, top_bottom_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, bottom_top_output, -1, bottom_top_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filtered_image = left_right_output+right_left_output+top_bottom_output+bottom_top_output;
    if (filtered_image.channels() == 3) cvtColor(filtered_image, filtered_image, COLOR_RGB2GRAY);

    // Freeing up memory
    left_right_output.deallocate(); right_left_output.deallocate(); top_bottom_output.deallocate(); bottom_top_output.deallocate();
    left_right_filter.deallocate(); right_left_filter.deallocate(); top_bottom_filter.deallocate(); bottom_top_filter.deallocate();

    GaussianBlur(filtered_image, filtered_image, Size(PreProcessing::BLUR_KERNEL_SIZE, PreProcessing::BLUR_KERNEL_SIZE), 0, 0);
    threshold(filtered_image, filtered_image, PreProcessing::THRESHOLD, 255, THRESH_BINARY);
    return filtered_image;
}

PreProcessing::PreProcessing(int blur_kernel_size, int threshold) {
    BLUR_KERNEL_SIZE = blur_kernel_size;
    THRESHOLD = threshold;
}


