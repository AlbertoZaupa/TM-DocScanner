#ifndef SERVER_APP_EDGE_CHASING_H
#define SERVER_APP_EDGE_CHASING_H

#endif

#include "opencv2/opencv.hpp"
#define W_E 0
#define E_W 1
#define N_S 2
#define S_N 3
#define KEEP_CHASING 0
#define LOOK_ASIDE_0 1
#define LOOK_ASIDE_1 2
#define LOOK_AHEAD 3
using namespace cv;

class PageFrame {
public:

    static int CHASE_DEPTH;
    static int TOLERANCE_FACTOR;
    static int ALLOW_SHIFT;

    explicit PageFrame(int chase_depth, int tolerance_factor, int allow_shift) {
        CHASE_DEPTH = chase_depth;
        TOLERANCE_FACTOR = tolerance_factor;
        ALLOW_SHIFT = allow_shift;
    }
};

Rect find_page_frame(const Mat &filtered_image);
bool edge_chase(const Mat &image, int row, int col, int chase_direction);
bool valid_pixel(const Mat &image, int row, int col);

void next_pixel_W_E(int &row, int &col);
void line_fit_W_E(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

void next_pixel_E_W(int &row, int &col);
void line_fit_E_W(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

void next_pixel_N_S(int &row, int &col);
void line_fit_N_S(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

void next_pixel_S_N(int &row, int &col);
void line_fit_S_N(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
