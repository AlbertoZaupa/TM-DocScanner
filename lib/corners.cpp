#include "corners.h"
#include <cstdlib>
#include <iostream>

void CornerCandidate::pick_col(CornerCandidate c1, CornerCandidate c2, int (*col_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.col_confidence && c2.col_confidence) {
        col_confidence = true;
        col = col_discriminating_func(c1.col, c2.col);
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        col_confidence = true;
        col = c1.col;
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        col_confidence = true;
        col = c2.col;
    }
    else if (c2.row_confidence && c1.row_confidence) {
        col_confidence = false;
        col = col_discriminating_func(c1.col, c2.col);
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        col_confidence = false;
        col = c2.col;
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        col_confidence = false;
        col = c1.col;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

void CornerCandidate::pick_row(CornerCandidate c1, CornerCandidate c2, int (*row_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.row_confidence && c2.row_confidence) {
        row_confidence = true;
        row = row_discriminating_func(c1.row, c2.row);
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        row_confidence = true;
        row = c1.row;
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        row_confidence = true;
        row = c2.row;
    }
    else if (c2.col_confidence && c1.col_confidence) {
        row_confidence = false;
        row = row_discriminating_func(c1.row, c2.row);
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        row_confidence = false;
        row = c2.row;
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        row_confidence = false;
        row = c1.row;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

void CornerCandidate::init(int col, int row, bool col_confidence, bool row_confidence) {
    this->col = col;
    this->row = row;
    this->col_confidence = col_confidence;
    this->row_confidence = row_confidence;
}

CornerCandidate::CornerCandidate(int col, int row, bool col_confidence, bool row_confidence) {
    this->col = col;
    this->row = row;
    this->col_confidence = col_confidence;
    this->row_confidence = row_confidence;
}
