#include "pipeline.h"
#include "binarization.h"
#include "page_frame.h"
#include "pre_processing.h"
#include "opencv2/opencv.hpp"
using namespace cv;

Mat execute_processing_pipeline(const Mat &input_image) {
    FILE* log = fopen("../benchmarks", "a");
    if (!log) {
        perror("Error opening benchmarks log file: ");
        exit(1);
    }
    // Pre processing
    Mat pre_processed_image = pre_process_image(input_image, log);

    auto start = std::chrono::system_clock::now();
    // Estrazione della cornice che contiene la pagina
    Rect page_frame = get_page_frame(pre_processed_image);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    fprintf(log, "%f\n", elapsed_seconds.count());

    start = std::chrono::system_clock::now();
    // Binarizzazione dell'immagine
    Mat binarized_image = StatisticsBasedBinarization::binarize_image(input_image(page_frame));
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    fprintf(log, "%f\n", elapsed_seconds.count());
    fclose(log);

    return binarized_image;
}
