#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <time.h>

#include "Plane.h"
#include "RANSAC.h"
#include "MatrixReaderWriter.h"

int main(int argc, char** argv)
{
    srand(time(NULL));


    //"garden.xyz"
    //"room.xyz"
    //"street.xyz"
    const char* points_path = "point clouds/street.xyz";
    const char* results_path = "results/street_result.txt";

    MatrixReaderWriter* mrw = new MatrixReaderWriter(points_path);
    int rowNum = mrw->rowNum;
    int columnNum = mrw->columnNum;
    std::cout << "Finished reading from file. Number of points is " << rowNum << std::endl;

    std::vector<cv::Point3d> points;
    cv::Point3d point;
    //Transfer mrw->data into points
    for (int i = 0; i < rowNum; i++) {
        point.x = mrw->data[columnNum * i];
        point.y = mrw->data[columnNum * i + 1];
        point.z = mrw->data[columnNum * i + 2];
        points.push_back(point);
    }

    Plane plane;

    std::vector<int> inliersIdx;
    LORANSAC_LSQ(points, inliersIdx, plane, 3, 0.05, 0.99, 10000);
    std::cout << "LORANSAC_LSQ is finished. " << inliersIdx.size() << " inlier/s was/were found" << std::endl;

    //Update data in mrw:
    double* data = new double[rowNum * 6];
    for (int i = 0; i < rowNum; i++) {
        data[6 * i + 0] = mrw->data[columnNum * i + 0]; //x   
        data[6 * i + 1] = mrw->data[columnNum * i + 1]; //y
        data[6 * i + 2] = mrw->data[columnNum * i + 2]; //z
        data[6 * i + 3] = 0;   //Red
        data[6 * i + 4] = 0;   //Green
        data[6 * i + 5] = 0;   //Blue
    }
    std::cout << "New data created." << std::endl;

    for (int idx = 0; idx < inliersIdx.size(); idx++) {
        int i = inliersIdx[idx];
        data[6 * i + 3] = 0;   //Red
        data[6 * i + 4] = 255; //Green
        data[6 * i + 5] = 0;   //Blue
    }
    std::cout << "Colors of inliers for new data is added." << std::endl;

    mrw->data = data;
    mrw->columnNum = 6;
    mrw->save(results_path);

    std::cout << "Finished." << std::endl;
    int k = cv::waitKey(0);

    return 0;
}