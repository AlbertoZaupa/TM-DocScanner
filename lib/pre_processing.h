#ifndef SERVER_APP_PRE_PROCESSING_H
#define SERVER_APP_PRE_PROCESSING_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

/*
 Questo modulo contiene il codice coinvolto nella fase di pre-processing.
 La classe PreProcessing contiene i parametri del modulo, ed esporta un costruttore per inizializzarne
 comodamente i valori.
*/

class PreProcessing {

public:
    static int BLUR_KERNEL_SIZE;
    static int THRESHOLD;
    static int HP_KERNEL_SIZE;

    explicit PreProcessing(int blur_kernel_size, int threshold);
};

Mat pre_process_image(const Mat &input_image, FILE* fh);
Mat edge_detection(const Mat &input_image);
