//
// Created by Alberto Zaupa on 07/07/22.
//

#include "pipeline.h"
#include "processing.h"
#include "utility.h"
#include "opencv2/opencv.hpp"
using namespace cv;

Mat ProcessingPipeline::execute() {
    return page_frame_detection_filtering_based(input_image);
}

Mat page_frame_detection_filtering_based(const Mat& input_image, int LARGE_BLUR_KERNEL_SIZE, int SMALL_BLUR_KERNEL_SIZE, int BIN_BLUR_TH, int EROSION_BLUR_TH) {
    Mat blurred_image, filtered_image, mask, mask_clone;
    GaussianBlur(input_image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    filtered_image = edge_detection(blurred_image);

    // The binarization step
    mask = filtered_image.clone();
    GaussianBlur(filtered_image, mask, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    mask_clone = mask.clone();
    threshold(mask_clone, mask, BIN_BLUR_TH, 255, THRESH_BINARY);
    Mat binarized_image = filtering_based_image_binarization(input_image, mask);

    // The erosion step
    GaussianBlur(filtered_image, mask, Size(SMALL_BLUR_KERNEL_SIZE, SMALL_BLUR_KERNEL_SIZE), 0, 0);
    mask_clone.deallocate();
    mask_clone = mask.clone();
    threshold(mask_clone, mask, EROSION_BLUR_TH, 255, THRESH_BINARY);
    Rect main_frame = erosion(mask);

    // Memory cleanup
    blurred_image.deallocate();
    filtered_image.deallocate();
    mask.deallocate();
    mask_clone.deallocate();

    return binarized_image(main_frame);
}

Mat page_frame_detection_stats_based(const Mat& input_image, int LARGE_BLUR_KERNEL_SIZE, int SMALL_BLUR_KERNEL_SIZE, int EROSION_BLUR_TH) {
    Mat blurred_image, filtered_image, mask, mask_clone;
    GaussianBlur(input_image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    filtered_image = edge_detection(blurred_image);

    // The image is binarized
    Mat binarized_image = stats_based_image_binarization(input_image);

    // The erosion step
    GaussianBlur(filtered_image, mask, Size(SMALL_BLUR_KERNEL_SIZE, SMALL_BLUR_KERNEL_SIZE), 0, 0);
    mask_clone = mask.clone();
    threshold(mask_clone, mask, EROSION_BLUR_TH, 255, THRESH_BINARY);
    Rect main_frame = erosion(mask);

    blurred_image.deallocate();
    filtered_image.deallocate();
    mask.deallocate();
    mask_clone.deallocate();

    return binarized_image(main_frame);
}

ProcessingPipeline::ProcessingPipeline(Mat input_image) {
    this->input_image = input_image;
}
