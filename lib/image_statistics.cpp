#include "image_statistics.h"
#include "opencv2/opencv.hpp"
using namespace cv;

/*
La funzione calcola, per ogni pixel dell'immagine, la media del valore in scala di grigio dei pixel
all'interno di una maschera quadrata, di lato block_size, centrata sul pixel in considerazione.

Sia N x M la dimensione dell'immagine e sia K la lunghezza del lato della maschera. Normalmente
calcolare la media all'interno della maschera per ogni pixel dell'immagine sarebbe un'operazione di 
complessità O(N x M x K x K). La funzione block_mean invece implementa un algoritmo di complessità
O(N x M).
*/

void block_mean(const Mat &m, unsigned char **mean_matrix, int block_size) {
    if (!block_size%2) {
        std::cerr<<"image_statistics.block_mean(): The value of the block size must be an odd number\n";
        exit(1);
    }
    int offset = block_size/2;
    int block_area = block_size*block_size;

    // Il funzionamento di base dell'algoritmo consiste nel tenere traccia, per ogni colonna,
    // della somma lungo le righe dei valori d'intensità di grigio dei pixel, in una finestra di lunghezza block_size.
    // Per calcolare la media all'interno della maschera centrata su un determinato pixel, è sufficiente sommare i
    // valori delle somme lungo le righe, per tutte colonne all'interno della maschera e dividere per l'area della
    // maschera. Il vantaggio di questo approccio è che per calcolare la media all'interno della maschera per il pixel
    // nella posizione (i, j), è sufficiente sottrare alla somma calcolata per il pixel nella posizione (i, j-1) il
    // valore della somma lungo le righe in posizione j - 1 - block_size/2, ed aggiungere il valore della somma 
    // lungo le righe alla posizione j + block_size/2, ed infine dividere per l'area della maschera. 
    long int block_rows_sum[m.size[1]];
    for (int i=0; i<m.size[1]; ++i) { // M operazioni
        block_rows_sum[i] = 0;
    }

    // Vengono inizializzati i valori delle somme lungo le righe
    int gray_value;
    for (int col=0; col<m.size[1]; col++) { // 2M x K operazioni
        for (int row=0; row<block_size; ++row) {
            gray_value = m.at<unsigned char>(row, col);
            block_rows_sum[col] += gray_value;
        }
    }

    long int moving_sum; float mean;
    for (int i=offset; i<m.size[0]-offset; ++i) { // N - K iterazioni
        if (i!=offset) {
            // Ogni volta che si passa alla righa successiva, bisogna aggiornare il valore delle somme lungo
            // le righe.
            for (int j=0; j<m.size[1]; ++j) { // 4M operazioni
                gray_value = m.at<unsigned char>(i-offset-1, j);
                block_rows_sum[j] -= gray_value;
                gray_value = m.at<unsigned char>(i+offset, j);
                block_rows_sum[j] += gray_value;
            }
        }

        // Il seguente ciclo for consiste in circa 4(M - K) + K operazioni
        for (int j=offset; j<m.size[1]-offset; ++j) { // M - K iterazioni
            if (j==offset) { // Questo blocco è eseguito 1 volta su M - K iterazioni
                moving_sum = 0;
                for (int k=0; k<block_size; ++k) {
                    moving_sum += block_rows_sum[k];
                }
            }
            else { // 4 operazioni
                moving_sum -= block_rows_sum[j-offset-1];
                moving_sum += block_rows_sum[j+offset];
            }
            mean = (float) moving_sum / (float) block_area;
            mean_matrix[i][j] = (unsigned char) mean;
        }
    }

    // Mettendo tutto insieme, il numero di operazioni effettuato dall'algoritmo è circa
    // (N - K)( 4(M - K) + K ) + (2M x K) + M . Dunque, al crescere di M, N e K, con K comunque molto
    // più piccolo di M ed N, la complessità dell'algoritmo è O(M x N).
}

/*
La seguente funzione per ogni pixel dell'immagine calcola, seguendo lo stesso algoritmo di block_mean, media e varianza
considerando i valori di grigio dei pixel all'interno di una maschera quadrata centrata nel pixel corrente.
Per guadagnare in efficienza tramite l'utilizzo delle somme parziali lungo le righe, bisogna calcolare la varianza come
var(x) = media(x^2) - (media(x))^2 .
*/

void block_stats(const Mat &m, unsigned char **mean_matrix, float **var_matrix, int block_size) {
    if (!block_size%2) {
        std::cerr<<"image_statistics.block_stats(): The value of the block size must be an odd number\n";
        exit(1);
    }
    int offset = block_size/2;
    int block_area = block_size*block_size;

    long int block_rows_sum[m.size[1]];
    long int block_rows_squares_sum[m.size[1]];
    for (int i=0; i<m.size[1]; ++i) {
        block_rows_sum[i] = 0;
        block_rows_squares_sum[i] = 0;
    }
    int grey_value;
    for (int col=0; col<m.size[1]; col++) {
        for (int row=0; row<block_size; ++row) {
            grey_value = m.at<unsigned char>(row, col);
            block_rows_sum[col] += grey_value;
            block_rows_squares_sum[col] += grey_value*grey_value;
        }
    }

    long int moving_sum, moving_squares_sum; float mean;
    for (int i=offset; i<m.size[0]-offset; ++i) {
        if (i!=offset) {
            for (int j=0; j<m.size[1]; ++j) {
                grey_value = m.at<unsigned char>(i-offset-1, j);
                block_rows_sum[j] -= grey_value;
                block_rows_squares_sum[j] -= grey_value*grey_value;
                grey_value = m.at<unsigned char>(i+offset, j);
                block_rows_sum[j] += grey_value;
                block_rows_squares_sum[j] += grey_value*grey_value;
            }
        }

        for (int j=offset; j<m.size[1]-offset; ++j) {
            if (j==offset) {
                moving_sum = 0;
                moving_squares_sum = 0;
                for (int k=0; k<block_size; ++k) {
                    moving_sum += block_rows_sum[k];
                    moving_squares_sum += block_rows_squares_sum[k];
                }
            }
            else {
                moving_sum -= block_rows_sum[j-offset-1];
                moving_sum += block_rows_sum[j+offset];
                moving_squares_sum -= block_rows_squares_sum[j-offset-1];
                moving_squares_sum += block_rows_squares_sum[j+offset];
            }
            mean = (float) moving_sum / (float) block_area;
            mean_matrix[i][j] = (unsigned char) mean;
            var_matrix[i][j] = ((float) moving_squares_sum / (float) block_area) - mean*mean;
        }
    }
}