//
// Created by Alberto Zaupa on 12/07/22.
//

#include "processing.h"
#include "utility.h"
#include "opencv2/opencv.hpp"

using namespace cv;

Rect erosion(const Mat &filtered_image, int EROSION_BOUNDARY) {
    // To simplify the erosion process, it is assumed that the borderds of the page frame are within the external image frame
    // of width IMAGE_WIDTH/2 and height IMAGE_HEIGHT/2
    int margin_search_x_bound = filtered_image.size[1] / 2;
    int margin_search_y_bound = filtered_image.size[0] / 2;

    // The erosion algorithm has aims to find the corners of the image. Its behavior can be pictured as a square moving 
    // from left to right, right to left, top to bottom and bottom to top, and stopping as soon as its leading side finds
    // a straight white line. 
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

Mat edge_detection(const Mat& input_image, int KERNEL_SIZE) {
    if (!KERNEL_SIZE%2) {
        std::cerr<<"processing.edge_detection(): The kernel size must be an odd number\n";
        exit(1);
    }

    // The filters' kernels are initialized
    Mat right_left_filter = Mat(1, KERNEL_SIZE, CV_32S);
    Mat left_right_filter = Mat(1, KERNEL_SIZE, CV_32S);
    Mat top_bottom_filter = Mat(KERNEL_SIZE, 1, CV_32S);
    Mat bottom_top_filter = Mat(KERNEL_SIZE, 1, CV_32S);
    //float filter_coefficient = (float) 2 / (float) KERNEL_SIZE; // each side of the filter calculates the mean of its side
    left_right_filter.forEach<int32_t>([KERNEL_SIZE] (int32_t &value, const int* p) -> void {
        if (p[1] < KERNEL_SIZE/2) value = -1;
        else if (p[1] > KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    right_left_filter.forEach<int32_t>([KERNEL_SIZE] (int32_t &value, const int* p) -> void {
        if (p[1] < KERNEL_SIZE/2) value = 1;
        else if (p[1] > KERNEL_SIZE/2) value = -1;
        else value = 0;
    });
    top_bottom_filter.forEach<int32_t>([KERNEL_SIZE] (int32_t &value, const int* p) -> void {
        if (p[0] < KERNEL_SIZE/2) value = -1;
        else if (p[0] > KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    bottom_top_filter.forEach<int32_t>([KERNEL_SIZE] (int32_t &value, const int* p) -> void {
        if (p[0] < KERNEL_SIZE/2) value = 1;
        else if (p[0] > KERNEL_SIZE/2) value = -1;
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

    return filtered_image;
}

Mat stats_based_image_binarization(const Mat &input_image, int BLOCK_SIZE, int CHUNK_SIZE, int CORRECTION_OFFSET) {
    Mat binarized_image = input_image.clone();
    if (binarized_image.channels() == 3) cvtColor(binarized_image, binarized_image, COLOR_RGB2GRAY);
    // For each pixel of the image the mean and variance of its neighbouring pixels are computed.
    // This statistics are computed on neighbourhoods of two different sizes.
    auto mean_matrix = new unsigned char*[input_image.size[0]], chunk_mean_matrix = new unsigned char*[input_image.size[0]];
    auto var_matrix = new float*[input_image.size[0]], chunk_var_matrix = new float*[input_image.size[0]];
    for (int i=0; i<input_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[input_image.size[1]];
        chunk_mean_matrix[i] = new unsigned char[input_image.size[1]];
        var_matrix[i] = new float[input_image.size[1]];
        chunk_var_matrix[i] = new float[input_image.size[1]];
    }
    int offset = CHUNK_SIZE/2;

    block_stats(binarized_image, mean_matrix, var_matrix, BLOCK_SIZE);
    block_stats(binarized_image, chunk_mean_matrix, chunk_var_matrix, CHUNK_SIZE);
    float var_th = mmean(var_matrix, offset, input_image.size[0]-offset, offset, input_image.size[1]-offset);

    binarized_image.forEach<unsigned char>([input_image, offset, var_th, chunk_mean_matrix, chunk_var_matrix, CORRECTION_OFFSET] (unsigned char &value, const int* position) -> void
    {
        if (position[0] < offset || position[1] < offset || position[0] >= input_image.size[0]-offset || position[1] >= input_image.size[1]-offset) {
            value = 255;
            return;
        }

        int y = position[0], x = position[1];

        if (chunk_var_matrix[y][x] < var_th) {
            value = 255;
        }
        else {
            value = value > chunk_mean_matrix[y][x] - CORRECTION_OFFSET ? 255 : 0;
        }
    }
    );

    for (int i=0; i<input_image.size[0]; ++i) {
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

Mat filtering_based_image_binarization(const Mat &input_image, const Mat &filtered_image, int BLOCK_SIZE, int CORRECTION_OFFSET) {
    Mat binarized_image = input_image.clone();
    if (binarized_image.channels() == 3) cvtColor(binarized_image, binarized_image, COLOR_RGB2GRAY);
    // For each pixel of the image the mean of the surrounding block is computed
    auto mean_matrix = new unsigned char*[input_image.size[0]];
    for (int i=0; i<input_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[input_image.size[1]];
        for (int j=0; j<input_image.size[1]; ++j) {
            mean_matrix[i][j] = 0;
        }
    }
    block_mean(binarized_image, mean_matrix, BLOCK_SIZE);

    // The binarization step
    binarized_image.forEach<unsigned char>([filtered_image, mean_matrix, BLOCK_SIZE, CORRECTION_OFFSET] (unsigned char &value, const int* p) -> void {
        int y = p[0], x = p[1];
        if (y <= BLOCK_SIZE/2 || x <= BLOCK_SIZE/2 || y >= filtered_image.size[0] - BLOCK_SIZE/2 || x >= filtered_image.size[1] - BLOCK_SIZE/2) {
            value = 255;
            return;
        }

        if (filtered_image.at<unsigned char>(y, x)) {
            value = value > mean_matrix[y][x] - CORRECTION_OFFSET ? 255 : 0;
        }
        else {
            value = 255;
        }
    });

    // Memory cleanup
    for (int i=0; i<input_image.size[0]; ++i) delete[] mean_matrix[i];
    delete[] mean_matrix;

    return binarized_image;
}