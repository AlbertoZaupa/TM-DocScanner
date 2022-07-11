//
// Created by Alberto Zaupa on 07/07/22.
//

#include "processing.h"
#include "utility.h"

Mat ProcessingPipeline::execute() {
    PageFrameDetectionFilteringBased detection = PageFrameDetectionFilteringBased(image);

    return detection.get_page_frame();
}

Mat PageFrameDetectionStatsBased::get_page_frame() {
    get_margin_search_bounds();
    Mat filtered_image = edge_detection(blurred_image);
    polish_filtered_image(filtered_image);
    working_slice = erosion(filtered_image, margin_search_y_bound, margin_search_x_bound);
    filtered_image.deallocate();
    Mat binarized_image = binarize_image();

    return binarized_image;
}

Mat PageFrameDetectionStatsBased::binarize_image() {
    Mat working_image = input_image(working_slice);
    // For each pixel of the image the mean and variance of its neighbouring pixels are computed.
    // This statistics are computed on neighbourhoods of two different sizes.
    return stats_based_image_binarization(working_image, BLOCK_SIZE, CHUNK_SIZE, SIGMA);
}

Mat PageFrameDetectionFilteringBased::get_page_frame() {
    Mat binarized_image = binarize_image();
    polish_filtered_image(filtered_image);
    get_margin_search_bounds();

    Mat output = binarized_image(erosion(filtered_image, margin_search_y_bound, margin_search_x_bound));
    binarized_image.deallocate();
    filtered_image.deallocate();
    return output;
}

Mat PageFrameDetectionFilteringBased::binarize_image() {
    // For each pixel of the image the mean of the surrounding block is computed
    auto mean_matrix = new unsigned char*[input_image.size[0]];
    for (int i=0; i<input_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[input_image.size[1]];
    }
    block_mean(input_image, mean_matrix, BLOCK_SIZE);

    // The filtering step is performed
    filtered_image = edge_detection(blurred_image);

    // The mask for the binarization process is computed
    Mat mask;
    GaussianBlur(filtered_image, mask, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
    threshold(mask.clone(), mask, LARGE_BLUR_POLISH_TH, 255, THRESH_BINARY);

    // The binarization step
    Mat binarized_image = input_image.clone();
    binarized_image.forEach<unsigned char>([mask, mean_matrix] (unsigned char &value, const int* p) -> void {
        unsigned char mask_value = mask.at<unsigned char>(p[0], p[1]);
        if (mask_value) {
            value = value < mean_matrix[p[0]][p[1]] - SIGMA / 2 ? 0 : 255;
        }
        else {
            value = 255;
        }
    });

    // Memory cleanup
    for (int i=0; i<input_image.size[0]; ++i) delete[] mean_matrix[i];
    delete[] mean_matrix;
    mask.deallocate();

    return binarized_image;
}

Mat stats_based_image_binarization(Mat working_image, int block_size, int chunk_size, int correction_offset) {
    // For each pixel of the image the mean and variance of its neighbouring pixels are computed.
    // This statistics are computed on neighbourhoods of two different sizes.
    auto mean_matrix = new unsigned char*[working_image.size[0]], chunk_mean_matrix = new unsigned char*[working_image.size[0]];
    auto var_matrix = new float*[working_image.size[0]], chunk_var_matrix = new float*[working_image.size[0]];
    for (int i=0; i<working_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[working_image.size[1]];
        chunk_mean_matrix[i] = new unsigned char[working_image.size[1]];
        var_matrix[i] = new float[working_image.size[1]];
        chunk_var_matrix[i] = new float[working_image.size[1]];
    }
    int offset = chunk_size/2;

    efficient_image_stats_calculation(working_image, mean_matrix, var_matrix, block_size);
    efficient_image_stats_calculation(working_image, chunk_mean_matrix, chunk_var_matrix, chunk_size);
    float var_th = mmean(var_matrix, offset, working_image.size[0]-offset, offset, working_image.size[1]-offset);


    Mat binarized_image = working_image.clone();
    binarized_image.forEach<unsigned char>([working_image, offset, var_th, chunk_mean_matrix, chunk_var_matrix, correction_offset] (unsigned char &value, const int* position) -> void
    {
        if (position[0] < offset || position[1] < offset || position[0] >= working_image.size[0]-offset || position[1] >= working_image.size[1]-offset) {
            value = 255;
            return;
        }

        int y = position[0], x = position[1];
        unsigned char chunk_mean = chunk_mean_matrix[y][x];
        float chunk_var = chunk_var_matrix[y][x];

        if (chunk_var < var_th) {
            value = 255;
        }
        else {
            value = value > chunk_mean - correction_offset ? 255 : 0;
        }
    }
    );

    for (int i=0; i<working_image.size[0]; ++i) {
        delete[] mean_matrix[i];
        delete[] chunk_mean_matrix[i];
        delete[] var_matrix[i];
        delete[] chunk_var_matrix[i];
    }
    delete[] mean_matrix;
    delete[] chunk_mean_matrix;
    delete[] var_matrix;
    delete[] chunk_var_matrix;

    return binarized_image;
}

Rect erosion(Mat filtered_image, int margin_search_y_bound, int margin_search_x_bound) {
    unsigned char boundary;
    int TL_corner[2], TR_corner[2], BL_corner[2], BR_corner[2];

    // Top Left corner search
    int X_stop[2] = {0, 0};
    bool found_margin = false;
    for (int row=0; row<margin_search_y_bound+EROSION_BOUNDARY && !found_margin; ++row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            boundary = 255;
            for (int k=row; k<row+EROSION_BOUNDARY && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    int Y_stop[2] = {0, 0};
    for (int col=0; col<margin_search_x_bound+EROSION_BOUNDARY && !found_margin; ++col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            boundary = 255;
            for (int k=col; k<col+EROSION_BOUNDARY && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    TL_corner[0] = min(X_stop[0], Y_stop[0]);
    TL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Top right corner search
    X_stop[0] = 0; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = 0; Y_stop[1] = filtered_image.size[1] - 1;
    found_margin = false;
    for (int row=0; row<margin_search_y_bound+EROSION_BOUNDARY && !found_margin; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1] - margin_search_x_bound - EROSION_BOUNDARY && !found_margin; --col) {
            boundary = 255;
            for (int k=row; k<row+EROSION_BOUNDARY && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col= filtered_image.size[1]-1; col>=filtered_image.size[1] - margin_search_x_bound - EROSION_BOUNDARY && !found_margin; --col) {
        for (int row=0; row<margin_search_y_bound+EROSION_BOUNDARY && !found_margin; ++row) {
            boundary = 255;
            for (int k=col; k>col-EROSION_BOUNDARY && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    TR_corner[0] = min(X_stop[0], Y_stop[0]);
    TR_corner[1] = max(X_stop[1], Y_stop[1]);

    // Bottom left corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = 0;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = 0;
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0] - margin_search_y_bound && !found_margin; --row) {
        for (int col=0; col<margin_search_x_bound+EROSION_BOUNDARY && !found_margin; ++col) {
            boundary = 255;
            for (int k=row; k>row-EROSION_BOUNDARY && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound+EROSION_BOUNDARY && !found_margin; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0] - margin_search_y_bound - EROSION_BOUNDARY && !found_margin; --row) {
            boundary = 255;
            for (int k=col; k<col + EROSION_BOUNDARY && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    BL_corner[0] = max(X_stop[0], Y_stop[0]);
    BL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Bottom right corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = filtered_image.size[1] - 1;
    found_margin = false;
    for (int row=filtered_image.size[0] - 1; row>=filtered_image.size[0] - margin_search_y_bound - EROSION_BOUNDARY && !found_margin; --row) {
        for (int col=filtered_image.size[1] - 1; col>=filtered_image.size[1] - margin_search_x_bound - EROSION_BOUNDARY && !found_margin; --col) {
            boundary = 255;
            for (int k=row; k>row - EROSION_BOUNDARY && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for(int col=filtered_image.size[1] - 1; col >= filtered_image.size[1] - margin_search_x_bound - EROSION_BOUNDARY && !found_margin; --col) {
        for (int row=filtered_image.size[0] - 1; row >= filtered_image.size[1] - margin_search_y_bound - EROSION_BOUNDARY && !found_margin; --row) {
            boundary = 255;
            for (int k=col; k>col - EROSION_BOUNDARY && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    BR_corner[0] = max(X_stop[0], Y_stop[0]);
    BR_corner[1] = max(X_stop[1], X_stop[1]);

    // The rectangle that represents the page frame
    int y_offset = min(TL_corner[0], TR_corner[0]);
    int x_offset = min(TL_corner[1], BL_corner[1]);

    int height = max(BL_corner[0], BR_corner[0]) - y_offset - 1;
    int width = max(TR_corner[1], BR_corner[1]) - x_offset - 1;
    return {x_offset, y_offset, width, height};
}

Mat edge_detection(Mat input_image) {
    // The filters' kernels are initialized
    Mat right_left_filter = Mat(1, HIGH_PASS_KERNEL_SIZE, CV_16S);
    Mat left_right_filter = Mat(1, HIGH_PASS_KERNEL_SIZE, CV_16S);
    Mat top_bottom_filter = Mat(HIGH_PASS_KERNEL_SIZE, 1, CV_16S);
    Mat bottom_top_filter = Mat(HIGH_PASS_KERNEL_SIZE, 1, CV_16S);
    left_right_filter.forEach<int16_t>([] (int16_t &value, const int* p) -> void {
        if (p[1] < HIGH_PASS_KERNEL_SIZE/2) value = -1;
        else if (p[1] > HIGH_PASS_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    right_left_filter.forEach<int16_t>([] (int16_t &value, const int* p) -> void {
        if (p[1] < HIGH_PASS_KERNEL_SIZE/2) value = 1;
        else if (p[1] > HIGH_PASS_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });
    top_bottom_filter.forEach<int16_t>([] (int16_t &value, const int* p) -> void {
        if (p[0] < HIGH_PASS_KERNEL_SIZE/2) value = -1;
        else if (p[0] > HIGH_PASS_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    bottom_top_filter.forEach<int16_t>([] (int16_t &value, const int* p) -> void {
        if (p[0] < HIGH_PASS_KERNEL_SIZE/2) value = 1;
        else if (p[0] > HIGH_PASS_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });

    // The matrices where the output of the filtering process will reside
    Mat right_left_output = Mat::zeros(input_image.size(), CV_16S);
    Mat left_right_output = Mat::zeros(input_image.size(), CV_16S);
    Mat top_bottom_output = Mat::zeros(input_image.size(), CV_16S);
    Mat bottom_top_output = Mat::zeros(input_image.size(), CV_16S);
    Mat filtered_image = Mat::zeros(input_image.size(), 0);

    // All 4 filters are applied
    filter2D(input_image, left_right_output, -1, left_right_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filtered_image.forEach<unsigned char>([left_right_output] (unsigned char &value, const int* p) -> void {
        value |= (unsigned char) left_right_output.at<unsigned char>(p[0], p[1]);
    });
    filter2D(input_image, right_left_output, -1, right_left_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filtered_image.forEach<unsigned char>([right_left_output] (unsigned char &value, const int* p) -> void {
        value |= (unsigned char) right_left_output.at<unsigned char>(p[0], p[1]);
    });
    filter2D(input_image, top_bottom_output, -1, top_bottom_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filtered_image.forEach<unsigned char>([top_bottom_output] (unsigned char &value, const int* p) -> void {
        value |= (unsigned char) top_bottom_output.at<unsigned char>(p[0], p[1]);
    });
    filter2D(input_image, bottom_top_output, -1, bottom_top_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filtered_image.forEach<unsigned char>([bottom_top_output] (unsigned char &value, const int* p) -> void {
        value |= (unsigned char) bottom_top_output.at<unsigned char>(p[0], p[1]);
    });

    // Freeing up memory
    left_right_output.deallocate(); right_left_output.deallocate(); top_bottom_output.deallocate(); bottom_top_output.deallocate();
    left_right_filter.deallocate(); right_left_filter.deallocate(); top_bottom_filter.deallocate(); bottom_top_filter.deallocate();

    return filtered_image;
}

void PageFrameDetection::polish_filtered_image(Mat filtered_image) {
    Mat src = filtered_image.clone();
    GaussianBlur(src, filtered_image, Size(SMALL_BLUR_KERNEL_SIZE, SMALL_BLUR_KERNEL_SIZE), 0, 0);
    src.deallocate();

    src = filtered_image.clone();
    threshold(src, filtered_image, SMALL_BLUR_POLISH_TH, 255, THRESH_BINARY);
    src.deallocate();
}

void PageFrameDetection::get_margin_search_bounds() {
    margin_search_x_bound = input_image.size[1] / 4;
    margin_search_y_bound = input_image.size[0] / 4;
}

ProcessingPipeline::ProcessingPipeline(Mat image) {
    this->image = image;
}

PageFrameDetectionStatsBased::PageFrameDetectionStatsBased(Mat image) {
    input_image = image;
    GaussianBlur(image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
}

PageFrameDetectionFilteringBased::PageFrameDetectionFilteringBased(Mat image) {
    input_image = image;
    GaussianBlur(image, blurred_image, Size(LARGE_BLUR_KERNEL_SIZE, LARGE_BLUR_KERNEL_SIZE), 0, 0);
}