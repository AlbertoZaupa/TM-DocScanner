#include "page_frame.h"
#include "opencv2/opencv.hpp"
#include "utility.h"
using namespace cv;

int PageFrame::CHASE_DEPTH = 400;
int PageFrame::TOLERANCE_FACTOR = 20;
int PageFrame::ALLOW_SHIFT = 4;

Rect find_page_frame(const Mat &filtered_image) {
    int margin_search_x_bound = filtered_image.size[1] / 2;
    int margin_search_y_bound = filtered_image.size[0] / 2;

    CornerCandidate TL_corner(0, 0, false, false);
    CornerCandidate TR_corner(filtered_image.size[1] - 1, 0, false, false);
    CornerCandidate BL_corner(0, filtered_image.size[0] - 1, false, false);
    CornerCandidate BR_corner(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);

    auto X_corner = new CornerCandidate(0, 0, false, false);
    auto Y_corner = new CornerCandidate(0, 0, false, false);
    bool found_margin = false;

    // Top Left corner search
    for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_corner->row = row;
                X_corner->col = col;
                X_corner->row_confidence = false;
                X_corner->col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_corner->row = row;
                Y_corner->col = col;
                Y_corner->row_confidence = true;
                Y_corner->col_confidence = false;
                found_margin = true;
            }
        }
    }
    CornerCandidate::pick_col(TL_corner, *X_corner, *Y_corner, min);
    CornerCandidate::pick_row(TL_corner, *X_corner, *Y_corner, min);
    delete X_corner;
    delete Y_corner;

    // Top right corner search
    X_corner = new CornerCandidate(filtered_image.size[1] - 1, 0, false, false);
    Y_corner = new CornerCandidate(filtered_image.size[1] - 1, 0, false, false);
    found_margin = false;
    for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_corner->row = row;
                X_corner->col = col;
                X_corner->row_confidence = false;
                X_corner->col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_corner->row = row;
                Y_corner->col = col;
                Y_corner->row_confidence = true;
                Y_corner->col_confidence = false;
                found_margin = true;
            }
        }
    }
    CornerCandidate::pick_col(TR_corner, *X_corner, *Y_corner, max);
    CornerCandidate::pick_row(TR_corner, *X_corner, *Y_corner, min);
    delete X_corner;
    delete Y_corner;

    // Bottom left corner search
    X_corner = new CornerCandidate(0, filtered_image.size[0] - 1, false, false);
    Y_corner = new CornerCandidate(0, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_corner->row = row;
                X_corner->col = col;
                X_corner->row_confidence = false;
                X_corner->col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_corner->row = row;
                Y_corner->col = col;
                Y_corner->row_confidence = true;
                Y_corner->col_confidence = false;
                found_margin = true;
            }
        }
    }
    CornerCandidate::pick_col(BL_corner, *X_corner, *Y_corner, min);
    CornerCandidate::pick_row(BL_corner, *X_corner, *Y_corner, max);
    delete X_corner;
    delete Y_corner;

    // Bottom right corner search
    X_corner = new CornerCandidate(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    Y_corner = new CornerCandidate(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_corner->row = row;
                X_corner->col = col;
                X_corner->row_confidence = false;
                X_corner->col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_corner->row = row;
                Y_corner->col = col;
                Y_corner->row_confidence = true;
                Y_corner->col_confidence = false;
                found_margin = true;
            }
        }
    }
    CornerCandidate::pick_col(BR_corner, *X_corner, *Y_corner, max);
    CornerCandidate::pick_row(BR_corner, *X_corner, *Y_corner, max);
    delete X_corner;
    delete Y_corner;

    CornerCandidate::pick_col(TL_corner, TL_corner, BL_corner, min);
    CornerCandidate::pick_col(BR_corner, BR_corner, TR_corner, max);
    CornerCandidate::pick_row(TL_corner, TL_corner, TR_corner, min);
    CornerCandidate::pick_row(BR_corner, BR_corner, BL_corner, max);

    // The rectangle that represents the page frame
    int height = BR_corner.row - TL_corner.row;
    int width = BR_corner.col - TL_corner.col;
    return {TL_corner.col, TL_corner.row, width, height};
}

bool edge_chase(const Mat &image, int row, int col, int chase_direction, std::vector<Point> &contour) {
    void (*next_pixel) (int &row, int &col);
    void (*line_fit) (float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
    void (*skip_ahead) (int &row, int &col, int skip);

    switch (chase_direction) {
        case W_E:
            next_pixel = next_pixel_W_E;
            line_fit = line_fit_W_E;
            skip_ahead = skip_ahead_W_E;
            break;
        case E_W:
            next_pixel = next_pixel_E_W;
            line_fit = line_fit_E_W;
            skip_ahead = skip_ahead_E_W;
            break;
        case N_S:
            next_pixel = next_pixel_N_S;
            line_fit = line_fit_N_S;
            skip_ahead = skip_ahead_N_S;
            break;
        case S_N:
            next_pixel = next_pixel_S_N;
            line_fit = line_fit_S_N;
            skip_ahead = skip_ahead_S_N;
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
                else {
                    skip_ahead(row, col, iterations + i -1);
                    return false;
                }

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

void skip_ahead_W_E(int &row, int &col, int skip) {
    col += skip;
}

void skip_ahead_E_W(int &row, int &col, int skip) {
    col -= skip;
}

void skip_ahead_N_S(int &row, int &col, int skip) {
    row += skip;
}

void skip_ahead_S_N(int &row, int &col, int skip) {
    row -= skip;
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

void CornerCandidate::pick_col(CornerCandidate &target, CornerCandidate &c1, CornerCandidate &c2,
                               int (*col_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.col_confidence && c2.col_confidence) {
        target.col_confidence = true;
        target.col = col_discriminating_func(c1.col, c2.col);
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        target.col_confidence = true;
        target.col = c1.col;
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        target.col_confidence = true;
        target.col = c2.col;
    }
    else if (c2.row_confidence && c1.row_confidence) {
        target.col_confidence = false;
        target.col = col_discriminating_func(c1.col, c2.col);
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        target.col_confidence = false;
        target.col = c2.col;
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        target.col_confidence = false;
        target.col = c1.col;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

void CornerCandidate::pick_row(CornerCandidate &target, CornerCandidate &c1, CornerCandidate &c2,
                               int (*row_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.row_confidence && c2.row_confidence) {
        target.row_confidence = true;
        target.row = row_discriminating_func(c1.row, c2.row);
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        target.row_confidence = true;
        target.row = c1.row;
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        target.row_confidence = true;
        target.row = c2.row;
    }
    else if (c2.col_confidence && c1.col_confidence) {
        target.row_confidence = false;
        target.row = row_discriminating_func(c1.row, c2.row);
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        target.row_confidence = false;
        target.row = c2.row;
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        target.row_confidence = false;
        target.row = c1.row;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

PageFrame::PageFrame(int chase_depth, int tolerance_factor, int allow_shift) {
    CHASE_DEPTH = chase_depth;
    TOLERANCE_FACTOR = tolerance_factor;
    ALLOW_SHIFT = allow_shift;
}

CornerCandidate::CornerCandidate(int col, int row, bool col_confidence, bool row_confidence) {
    this->col = col;
    this->row = row;
    this->col_confidence = col_confidence;
    this->row_confidence = row_confidence;
}