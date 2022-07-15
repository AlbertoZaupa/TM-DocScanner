#ifndef SERVER_APP_CORNERS_H
#define SERVER_APP_CORNERS_H

#endif

class CornerCandidate {
public:
    int col, row;
    bool col_confidence, row_confidence;

    explicit CornerCandidate(int col, int row, bool col_confidence, bool row_confidence);
    void init(int col, int row, bool col_confidence, bool row_confidence);
    void pick_col(CornerCandidate c1, CornerCandidate c2, int (*col_discriminating_func) (int, int));
    void pick_row(CornerCandidate c1, CornerCandidate c2, int (*row_discriminating_func) (int, int));
};
