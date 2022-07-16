#ifndef SERVER_APP_BINARIZATION_H
#define SERVER_APP_BINARIZATION_H

#endif

#include "opencv2/opencv.hpp"
using namespace cv;

/*
 Questo modulo contiene il codice per effettuare la binarizzazione dell'immagine. Sono presenti due classi, ognuna
 delle quali contiene dei parametri, un costruttore per inizializzarli, ed una funzione binarize_image che realizza
 la binarizzazione dell'immagine.
 Le due classi sono rappresentative di due possibili approcci alla binarizzazione, uno basato esclusivamente sull'
 estrazione di media e varianza dell'immagine ed uno che sfrutta anche dei filtri passa alto.
*/

class StatisticsBasedBinarization {
public:
    static int BLOCK_SIZE;
    static int CHUNK_SIZE;
    static int CORRECTION_OFFSET;

    explicit StatisticsBasedBinarization(int block_size, int chunk_size, int correction_offset);
    static Mat binarize_image(const Mat &input_image);
};

class FilteringBasedBinarization {
public:
    static int BLOCK_SIZE;
    static int CORRECTION_OFFSET;
    static int BLUR_KERNEL_SIZE;
    static int THRESHOLD;

    explicit FilteringBasedBinarization(int block_size, int correction_offset, int blur_kernel_size, int threshold,
                                        int hp_kernel_size);
    static Mat binarize_image(const Mat &input_image);
};