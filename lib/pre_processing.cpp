#include "pre_processing.h"
#include "opencv2/opencv.hpp"

int PreProcessing::BLUR_KERNEL_SIZE = 51;
int PreProcessing::THRESHOLD = 30;
int PreProcessing::HP_KERNEL_SIZE = 11;

/*
 Questa funzione mette insieme i passaggi che costituiscono la fase di pre-processing, il cui scopo
 è preparare l'immagine per l'elaborazione successiva, ovvero l'estrazione della pagina.
*/

Mat pre_process_image(const Mat &input_image) {

    Mat output_image = input_image.clone();

    // L'immagine viene filtrata tramite un filtro mediano ad ampia maschera. Questo passaggio, che ha lo scopo
    // di rimuovere dall'immagine le variazioni locali, mantenendo il più possibile evidenti i punti di bordo tra
    // gli oggetti dell'immagine, determina pesantemente l'efficacia dell'estrazione della pagina.
    // Il filtro mediano sfuoca pesantemente il testo scritto all'interno del foglio scannerizzato ed il rumore
    // di bordo, mentre mantiene abbastanza evidenti i bordi del foglio scannerizzato.
    medianBlur(output_image, output_image, PreProcessing::BLUR_KERNEL_SIZE);

    // Il risultato viene filtrato tramite dei passa-alto per evidenziare i bordi dell'immagine.
    output_image = edge_detection(output_image);

    return output_image;
}

/*
 La seguente funzione estrae i bordi dell'immagine. Questo passaggio consiste in un filtraggio tramite passa-alto,
 realizzato in 4 passaggi: sinistra -> destra, destra -> sinistra, alto -> basso, basso -> alto.
 La maschera base del kernel è [ -1 0 1 ].
*/

Mat edge_detection(const Mat &input_image) {
    // Le maschere dei filtri vengono inizializzate
    Mat right_left_filter = Mat(1, PreProcessing::HP_KERNEL_SIZE, CV_32S);
    Mat left_right_filter = Mat(1, PreProcessing::HP_KERNEL_SIZE, CV_32S);
    Mat top_bottom_filter = Mat(PreProcessing::HP_KERNEL_SIZE, 1, CV_32S);
    Mat bottom_top_filter = Mat(PreProcessing::HP_KERNEL_SIZE, 1, CV_32S);

    // Sia N la lunghezza del filtro, con N dispari. I primi N/2 coefficienti sono pari a -1,
    // il coefficiente centrale è pari a 0, ed i successivi N/2 sono pari ad 1.
    left_right_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[1] < PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else if (p[1] > PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    right_left_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[1] < PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else if (p[1] > PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });
    top_bottom_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[0] < PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else if (p[0] > PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else value = 0;
    });
    bottom_top_filter.forEach<int32_t>([] (int32_t &value, const int* p) -> void {
        if (p[0] < PreProcessing::HP_KERNEL_SIZE/2) value = 1;
        else if (p[0] > PreProcessing::HP_KERNEL_SIZE/2) value = -1;
        else value = 0;
    });

    // Le matrici contenenti i risultati delle convoluzioni
    Mat right_left_output = Mat::zeros(input_image.size(), CV_32F);
    Mat left_right_output = Mat::zeros(input_image.size(), CV_32F);
    Mat top_bottom_output = Mat::zeros(input_image.size(), CV_32F);
    Mat bottom_top_output = Mat::zeros(input_image.size(), CV_32F);
    Mat filtered_image = Mat::zeros(input_image.size(), 0);

    // I filtri vengono applicati
    filter2D(input_image, left_right_output, -1, left_right_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, right_left_output, -1, right_left_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, top_bottom_output, -1, top_bottom_filter, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(input_image, bottom_top_output, -1, bottom_top_filter, Point(-1, -1), 0, BORDER_DEFAULT);

    // I risultati delle convoluzioni sono tra loro sommati
    filtered_image = left_right_output+right_left_output+top_bottom_output+bottom_top_output;
    if (filtered_image.channels() == 3) cvtColor(filtered_image, filtered_image, COLOR_RGB2GRAY);

    // Le matrici non più necessarie sono deallocate
    left_right_output.deallocate(); right_left_output.deallocate(); top_bottom_output.deallocate(); bottom_top_output.deallocate();
    left_right_filter.deallocate(); right_left_filter.deallocate(); top_bottom_filter.deallocate(); bottom_top_filter.deallocate();

    // Il risultato viene filtrato tramite un passabasso, e successivamente binarizzato applicando una soglia.
    // Queste due operazioni hanno l'effetto di ripulire l'immagine filtrata da "falsi" bordi, e di inspessire i bordi
    // reali.
    GaussianBlur(filtered_image, filtered_image, Size(PreProcessing::BLUR_KERNEL_SIZE, PreProcessing::BLUR_KERNEL_SIZE), 0, 0);
    threshold(filtered_image, filtered_image, PreProcessing::THRESHOLD, 255, THRESH_BINARY);

    return filtered_image;
}

PreProcessing::PreProcessing(int blur_kernel_size, int threshold) {
    BLUR_KERNEL_SIZE = blur_kernel_size;
    THRESHOLD = threshold;
}


