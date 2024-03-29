#include "binarization.h"
#include "image_statistics.h"
#include "utility.h"
#include "pre_processing.h"
#include "opencv2/opencv.hpp"
using namespace cv;

int StatisticsBasedBinarization::BLOCK_SIZE = 9;
int StatisticsBasedBinarization::CHUNK_SIZE = 37;
int StatisticsBasedBinarization::CORRECTION_OFFSET = 10;

int FilteringBasedBinarization::BLOCK_SIZE = 19;
int FilteringBasedBinarization::CORRECTION_OFFSET = 10;
int FilteringBasedBinarization::BLUR_KERNEL_SIZE = 51;
int FilteringBasedBinarization::THRESHOLD = 10;

/*
 La seguente funzione binarizza un immagine sulla base delle realtive statistiche locali. Come primo passaggio viene 
 calcolata una maschera che identifica all'interno dell'immagine le regioni contenenti testo scritto, confrontando
 i valori delle varianze locali calcolati su maschere di dimensioni diverse. Successivamente
 all'interno di tale maschera vengono anneriti i pixel il cui valore è minore della media locale.
 L'ipotesi alla base di questo approccio è che nelle zone dell'immagine in cui è contenuto testo scritto 
 la varianza locale è maggiore della varianza globale.
*/

Mat StatisticsBasedBinarization::binarize_image(const Mat &input_image) {
    Mat binarized_image = input_image.clone();
    if (binarized_image.channels() == 3) cvtColor(binarized_image, binarized_image, COLOR_RGB2GRAY);

    // Vengono inizializzate le matrici che contengono le statistiche locali dell'immagine. Tali statistiche sono
    // calcolate su una maschera più piccola, chiamata BLOCK, e su una maschera più grande, chiamata CHUNK.
    auto mean_matrix = new unsigned char*[input_image.size[0]], chunk_mean_matrix = new unsigned char*[input_image.size[0]];
    auto var_matrix = new float*[input_image.size[0]], chunk_var_matrix = new float*[input_image.size[0]];
    for (int i=0; i<input_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[input_image.size[1]];
        chunk_mean_matrix[i] = new unsigned char[input_image.size[1]];
        var_matrix[i] = new float[input_image.size[1]];
        chunk_var_matrix[i] = new float[input_image.size[1]];
    }
    int offset = CHUNK_SIZE/2;

    // Vengono calcolate le statistiche locali dell'immagine
    block_stats(binarized_image, mean_matrix, var_matrix, BLOCK_SIZE);
    block_stats(binarized_image, chunk_mean_matrix, chunk_var_matrix, CHUNK_SIZE);

    // Per determinare se un pixel appartiene ad una regione dell'immagine dove è presente del testo, il valore della
    // relativa varianza locale, calcolata all'interno della maschera di dimensione più grande, viene confrontato con
    // la media delle varianze locali calcolate all'interno della maschera più piccola. Se la varianza locale è maggiore
    // della media delle varianze locali, il pixel fa parte di una regione contenente del testo.
    float var_th = mmean(var_matrix, offset, input_image.size[0]-offset, offset, input_image.size[1]-offset);

    binarized_image.forEach<unsigned char>([input_image, offset, var_th, chunk_mean_matrix, chunk_var_matrix, mean_matrix, var_matrix] (unsigned char &value, const int* position) -> void
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

/*
La seguente funzione implementa la binarizzazione dell'immagine utilizzando dei filtri passa-alto. I filtri utilizzati
sono gli stessi che vengono applicati durante il pre-processing per esaltare le regioni di bordo.
Il risultato dell'applicazione dei filtri è una maschera che identifica le regioni dove è presente del testo scritto. Per
completare la binarizzazione, il valore di grigio dei pixel nelle zone che contengono del testo viene confrontato con
la media locale, se l'intensità di grigio del pixel è minore della media viene portato a 0, altrimenti a 255. 
*/

Mat FilteringBasedBinarization::binarize_image(const Mat &input_image) {
    Mat binarized_image = input_image.clone();
    if (binarized_image.channels() == 3) cvtColor(binarized_image, binarized_image, COLOR_RGB2GRAY);
    auto mean_matrix = new unsigned char*[input_image.size[0]];
    for (int i=0; i<input_image.size[0]; ++i) {
        mean_matrix[i] = new unsigned char[input_image.size[1]];
        for (int j=0; j<input_image.size[1]; ++j) {
            mean_matrix[i][j] = 0;
        }
    }
    block_mean(binarized_image, mean_matrix, BLOCK_SIZE);

    Mat mask = input_image.clone();
    GaussianBlur(mask, mask, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0, 0);
    mask = edge_detection(mask);

    binarized_image.forEach<unsigned char>([mask, mean_matrix] (unsigned char &value, const int* p) -> void {
        int y = p[0], x = p[1];
        if (y <= BLOCK_SIZE/2 || x <= BLOCK_SIZE/2 || y >= mask.size[0] - BLOCK_SIZE/2 || x >= mask.size[1] - BLOCK_SIZE/2) {
            value = 255;
            return;
        }

        if (mask.at<unsigned char>(y, x)) {
            value = value > mean_matrix[y][x] - CORRECTION_OFFSET ? 255 : 0;
        }
        else {
            value = 255;
        }
    });

    for (int i=0; i<input_image.size[0]; ++i) delete[] mean_matrix[i];
    delete[] mean_matrix;
    mask.deallocate();

    return binarized_image;
}

StatisticsBasedBinarization::StatisticsBasedBinarization(int block_size, int chunk_size, int correction_offset) {
    BLOCK_SIZE = block_size;
    CHUNK_SIZE = chunk_size;
    CORRECTION_OFFSET = correction_offset;
}

FilteringBasedBinarization::FilteringBasedBinarization(int block_size, int correction_offset, int blur_kernel_size,
                                                       int threshold, int hp_kernel_size) {
    BLOCK_SIZE = block_size;
    CORRECTION_OFFSET = correction_offset;
    BLUR_KERNEL_SIZE = blur_kernel_size;
    THRESHOLD = threshold;
}

