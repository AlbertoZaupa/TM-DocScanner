#include "pipeline.h"
#include "processing.h"
#include "utility.h"
#include "page_frame.h"
#include "opencv2/opencv.hpp"
using namespace cv;

Mat ProcessingPipeline::filtering_approach(const Mat& input_image) {
    Mat blurred_image, filtered_image, mask;
    GaussianBlur(input_image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    filtered_image = edge_detection(blurred_image, EDGE_DET_KERNEL_LENGTH);

    // The binarization step
    mask = filtered_image.clone();
    GaussianBlur(filtered_image, mask, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    threshold(mask, mask, BIN_BLUR_TH, 255, THRESH_BINARY);
    Mat binarized_image = filtering_based_image_binarization(input_image, mask, FILTERING_BLOCK_SIZE, CORRECTION_OFFSET);

    // The erosion step
    GaussianBlur(filtered_image, mask, Size(SMALL_BLUR_KERNEL_SIZE, SMALL_BLUR_KERNEL_SIZE), 0, 0);
    threshold(mask, mask, EROSION_BLUR_TH, 255, THRESH_BINARY);
    Rect main_frame = find_page_frame(mask);//erosion(mask, EROSION_BOUNDARY);

    // Memory cleanup
    blurred_image.deallocate();
    filtered_image.deallocate();
    mask.deallocate();

    return binarized_image(main_frame);
}

Mat ProcessingPipeline::statistical_approach(const Mat& input_image) {
    Mat blurred_image, filtered_image, mask;
    GaussianBlur(input_image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    filtered_image = edge_detection(blurred_image, EDGE_DET_KERNEL_LENGTH);

    // The image is binarized
    Mat binarized_image = stats_based_image_binarization(input_image, STATS_BLOCK_SIZE, STATS_CHUNK_SIZE, CORRECTION_OFFSET);

    // The erosion step
    GaussianBlur(filtered_image, mask, Size(SMALL_BLUR_KERNEL_SIZE, SMALL_BLUR_KERNEL_SIZE), 0, 0);
    threshold(mask, mask, EROSION_BLUR_TH, 255, THRESH_BINARY);
    Rect main_frame = erosion(mask, EROSION_BOUNDARY);

    blurred_image.deallocate();
    filtered_image.deallocate();
    mask.deallocate();

    return binarized_image(main_frame);
}

ProcessingPipeline::ProcessingPipeline(int large_blur_kernel_size, int small_blur_kernel_size, int bin_blur_th,
                                       int erosion_blur_th, int stats_block_size, int stats_chunk_size,
                                       int filtering_block_size, int edge_det_kernel_length, int correction_offset,
                                       int erosion_boundary) {
    LARGE_BLUR_KERNEL_SIZE = large_blur_kernel_size;
    SMALL_BLUR_KERNEL_SIZE = small_blur_kernel_size;
    BIN_BLUR_TH = bin_blur_th;
    EROSION_BLUR_TH = erosion_blur_th;
    STATS_BLOCK_SIZE = stats_block_size;
    STATS_CHUNK_SIZE = stats_chunk_size;
    FILTERING_BLOCK_SIZE = filtering_block_size;
    EDGE_DET_KERNEL_LENGTH = edge_det_kernel_length;
    CORRECTION_OFFSET = correction_offset;
    EROSION_BOUNDARY = erosion_boundary;
}
