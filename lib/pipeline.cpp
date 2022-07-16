#include "pipeline.h"
#include "binarization.h"
#include "page_frame.h"
#include "pre_processing.h"
#include "opencv2/opencv.hpp"
using namespace cv;

Mat execute_processing_pipeline(const Mat &input_image) {
    // Pre processing
    Mat pre_processed_image = pre_process_image(input_image);

    // Estrazione della cornice che contiene la pagina
    Rect page_frame = get_page_frame(input_image, pre_processed_image);

    // Binarizzazione dell'immagine
    Mat binarized_image = StatisticsBasedBinarization::binarize_image(input_image(page_frame));

    return binarized_image;
}
