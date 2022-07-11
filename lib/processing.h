#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif //SERVER_APP_PROCESSING_H

#include "opencv2/opencv.hpp"
#define BLOCK_SIZE 9
#define CHUNK_SIZE 37 // 4 x BLOCK_SIZE + 1
#define SIGMA 10 // Experimentally assigned value
#define EROSION_BOUNDARY 100
#define HIGH_PASS_KERNEL_SIZE 11
#define LARGE_BLUR_KERNEL_SIZE 51
#define LARGE_BLUR_POLISH_TH 10
#define SMALL_BLUR_KERNEL_SIZE 5
#define SMALL_BLUR_POLISH_TH 30
using namespace cv;

class ProcessingPipeline {
    Mat image;
public:
    explicit ProcessingPipeline(Mat image);
    Mat execute();
};

class PageFrameDetection {
    virtual Mat binarize_image() = 0;
protected:
    Mat input_image, blurred_image;
    int margin_search_x_bound, margin_search_y_bound;

    static void polish_filtered_image(Mat filtered_image);
    void get_margin_search_bounds();
public:
    virtual Mat get_page_frame() = 0;
};

class PageFrameDetectionStatsBased : public PageFrameDetection {
    Rect working_slice;

    Mat binarize_image() override;
public:
    explicit PageFrameDetectionStatsBased(Mat image);
    Mat get_page_frame() override;
};

class PageFrameDetectionFilteringBased : PageFrameDetection {
    Mat filtered_image;

    Mat binarize_image() override;
public:
    explicit PageFrameDetectionFilteringBased(Mat image);
    Mat get_page_frame() override;
};

Rect erosion(Mat filtered_image, int margin_search_x_bound, int margin_search_y_bound);
Mat stats_based_image_binarization(Mat working_image, int block_size, int chunk_size, int correction_offset);
Mat edge_detection(Mat input_image);


