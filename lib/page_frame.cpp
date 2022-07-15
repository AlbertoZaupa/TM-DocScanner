#include "page_frame.h"
#include "opencv2/opencv.hpp"
using namespace cv;

int PageFrame::CHASE_DEPTH = 200;
int PageFrame::TOLERANCE_FACTOR = 20;
int PageFrame::ALLOW_SHIFT = 4;

Rect find_page_frame(const Mat &filtered_image) {
    int margin_search_x_bound = filtered_image.size[1] / 2;
    int margin_search_y_bound = filtered_image.size[0] / 2;

    int TL_corner[2], TR_corner[2], BL_corner[2], BR_corner[2];
    int X_stop[2] = {0, 0}, Y_stop[2] = {0, 0};

    // Top Left corner search
    for (int row=0; row<margin_search_y_bound; ++row) {
        for (int col=0; col<margin_search_x_bound; ++col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_stop[0] = row;
                X_stop[1] = col;
                break;
            }
        }
    }
    for (int col=0; col<margin_search_x_bound; ++col) {
        for (int row=0; row<margin_search_y_bound; ++row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                break;
            }
        }
    }
    TL_corner[0] = min(X_stop[0], Y_stop[0]);
    TL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Top right corner search
    X_stop[0] = 0; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = 0; Y_stop[1] = filtered_image.size[1] - 1;
    for (int row=0; row<margin_search_y_bound; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound; --col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_stop[0] = row;
                X_stop[1] = col;
                break;
            }
        }
    }
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound; --col) {
        for (int row=0; row<margin_search_y_bound; ++row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                break;
            }
        }
    }
    TR_corner[0] = min(X_stop[0], Y_stop[0]);
    TR_corner[1] = max(X_stop[1], Y_stop[1]);

    // Bottom left corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = 0;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = 0;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound; --row) {
        for (int col=0; col<margin_search_x_bound; ++col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_stop[0] = row;
                X_stop[1] = col;
                break;
            }
        }
    }
    for (int col=0; col<margin_search_x_bound; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound; --row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                break;
            }
        }
    }
    BL_corner[0] = max(X_stop[0], Y_stop[0]);
    BL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Bottom right corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = filtered_image.size[1] - 1;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound; --row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound; --col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_stop[0] = row;
                X_stop[1] = col;
                break;
            }
        }
    }
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound; --col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound; --row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                break;
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

bool edge_chase(const Mat &image, int row, int col, int chase_direction) {
    void (*next_pixel) (int &row, int &col);
    void (*line_fit) (float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

    switch (chase_direction) {
        case W_E:
            next_pixel = next_pixel_W_E;
            line_fit = line_fit_W_E;
            break;
        case E_W:
            next_pixel = next_pixel_E_W;
            line_fit = line_fit_E_W;
            break;
        case N_S:
            next_pixel = next_pixel_N_S;
            line_fit = line_fit_N_S;
            break;
        case S_N:
            next_pixel = next_pixel_S_N;
            line_fit = line_fit_S_N;
            break;
        default:
            std::cerr<<"edge_chasing.edge_chase(): Possible directions are West -> East, East -> West, North -> South, South -> North\n";
            exit(1);
    }

    // If the first pixel does not belong to a contour, the chase stops
    unsigned char gray_value;
    gray_value = image.at<unsigned char>(row, col);
    if (!gray_value) return false;

    int start_row = row;
    int start_col = col;
    next_pixel(row, col);

    // The state variables are initialized
    int iterations = 1;
    int last_shift_iteration = 0;
    int current_state;
    int next_state = KEEP_CHASING;

    // The state machine
    for (;;) {
        current_state = next_state;

        switch (current_state) {
            case KEEP_CHASING:
                next_pixel(row, col);
                if (!valid_pixel(image, row, col)) return true;

                gray_value = image.at<unsigned char>(row, col);
                // The next state is computed
                if (gray_value) {
                    iterations++;
                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else if (iterations - last_shift_iteration >= PageFrame::ALLOW_SHIFT) {
                    switch (chase_direction) {
                        case W_E:
                        case E_W:
                            if (valid_pixel(image, row, col + 1)) next_state = LOOK_ASIDE_0;
                            else next_state = LOOK_ASIDE_1;
                            break;
                        case N_S:
                        case S_N:
                            if (valid_pixel(image, row + 1, col)) next_state = LOOK_ASIDE_0;
                            else next_state = LOOK_ASIDE_1;
                    }
                }
                else next_state = LOOK_AHEAD;

                break;
            case LOOK_ASIDE_0:
                if (chase_direction == W_E || chase_direction == E_W)
                    gray_value = image.at<unsigned char>(row, col + 1);
                else
                    gray_value = image.at<unsigned char>(row + 1, col);

                // The next state is computed
                if (gray_value) {
                    last_shift_iteration = iterations;
                    iterations++;
                    if (chase_direction == W_E || chase_direction == E_W) ++col;
                    else ++row;

                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else {
                    switch (chase_direction) {
                        case W_E:
                        case E_W:
                            if (valid_pixel(image, row, col - 1)) next_state = LOOK_ASIDE_1;
                            else next_state = LOOK_AHEAD;
                            break;
                        case N_S:
                        case S_N:
                            if (valid_pixel(image, row - 1, col)) next_state = LOOK_ASIDE_1;
                            else next_state = LOOK_AHEAD;
                    }
                }

                break;
            case LOOK_ASIDE_1:
                if (chase_direction == W_E || chase_direction == E_W)
                    gray_value = image.at<unsigned char>(row, col - 1);
                else
                    gray_value = image.at<unsigned char>(row - 1, col);

                // The next state is computed
                if (gray_value) {
                    last_shift_iteration = iterations;
                    iterations++;
                    if (chase_direction == W_E || chase_direction == E_W) --col;
                    else --row;

                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else next_state = LOOK_AHEAD;

                break;
            case LOOK_AHEAD:
                // The coefficient of the line to chase along are computed
                float M;
                if (chase_direction == W_E || chase_direction == E_W)
                    M = float (row - start_row) / float (col - start_col);
                else M = float (col - start_col) / float (row - start_row);

                int current_row;
                int current_col;
                current_row = row;
                current_col = col;
                int projected_row, projected_col;
                int i;
                for (i=1; i<PageFrame::CHASE_DEPTH/PageFrame::TOLERANCE_FACTOR && !gray_value; ++i) {
                    if (iterations + i >= PageFrame::CHASE_DEPTH) return true;

                    next_pixel(current_row, current_col);
                    line_fit(M, row, col, current_row, current_col, projected_row, projected_col);
                    if (!valid_pixel(image, projected_row, projected_col)) return true;
                    gray_value = image.at<unsigned char>(projected_row, projected_col);
                }

                // The future state is computed
                if (i < PageFrame::CHASE_DEPTH/PageFrame::TOLERANCE_FACTOR || gray_value) {
                    iterations += i;
                    row = projected_row;
                    col = projected_col;
                    next_state = KEEP_CHASING;
                }
                else return false;

                break;
            default:
                std::cerr<<"edge_chasing.edge_chase(): invalid state code "<<current_state<<"\n";
                exit(1);
        }
    }
}

void next_pixel_W_E(int &row, int &col) {
    col += 1;
}

void next_pixel_E_W(int &row, int &col) {
    col -= 1;
}

void next_pixel_N_S(int &row, int &col) {
    row += 1;
}

void next_pixel_S_N(int &row, int &col) {
    row -= 1;
}

void line_fit_W_E(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_col - start_col;
    int dy = M*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_E_W(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = start_col - curr_col;
    int dy = (-M)*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_N_S(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_row - start_row;
    int dy = M*dx;

    projected_row = curr_row;
    projected_col = start_col + dy;
}

void line_fit_S_N(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = start_row - curr_row;
    int dy = (-M)*dx;

    projected_row = curr_row;
    projected_col = start_col + dy;
}

bool valid_pixel(const Mat &image, int row, int col) {
    return row < image.size[0] && col < image.size[1];
}