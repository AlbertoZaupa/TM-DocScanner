#ifndef SERVER_APP_PROCESSING_H
#define SERVER_APP_PROCESSING_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

/*
Questo modulo esporta una funzione che mette insieme i vari passaggi della pipeline di elaborazione dell'immagine
*/

Mat execute_processing_pipeline(const Mat &input_image);
