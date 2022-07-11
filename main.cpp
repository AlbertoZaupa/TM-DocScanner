#include <iostream>
#include "lib/processing.h"
#include "opencv2/opencv.hpp"
#include "lib/utility.h"

using namespace std; using namespace cv;
int main() {
    Mat img = imread("../images/good_enough.jpg", IMREAD_GRAYSCALE);
    ProcessingPipeline pipeline = ProcessingPipeline(img);
    imshow("Processed image", pipeline.execute());
    waitKey(0);
    return 0;
}
