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

/*
 Questo modulo contiene il codice responsabile di rimuovere lo sfondo dall'immagine, mantenendo solo il rettangolo che
 contiene il foglio da scannerizzare. Lo sfondo solitamente corrisponde al tavolo su cui è appoggiato il foglio.
 La classe PageFrame contiene esclusivamente dei parametri, che possono essere inizializzati tramite un apposito
 costruttore.
*/

class PageFrame {
public:

    static int CHASE_DEPTH;
    static int TOLERANCE_FACTOR;
    static int ALLOW_SHIFT;
    static int RUDIMENTARY_DEPTH;

    explicit PageFrame(int chase_depth, int tolerance_factor, int allow_shift);
};

Rect get_page_frame(const Mat &base_image, const Mat &filtered_image);
Rect rudimentary_get_page_frame(const Mat &filtered_image);
bool edge_chase(const Mat &image, int row, int col, int chase_direction, std::vector<Point> &contour);
bool valid_pixel(const Mat &image, int row, int col);

void next_pixel_W_E(int &row, int &col);
void line_fit_W_E(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
void skip_ahead_W_E (int &row, int &col, int skip);

void next_pixel_E_W(int &row, int &col);
void line_fit_E_W(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
void skip_ahead_E_W (int &row, int &col, int skip);

void next_pixel_N_S(int &row, int &col);
void line_fit_N_S(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
void skip_ahead_N_S (int &row, int &col, int skip);

void next_pixel_S_N(int &row, int &col);
void line_fit_S_N(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);
void skip_ahead_S_N (int &row, int &col, int skip);
