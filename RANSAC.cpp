#include "RANSAC.h"
#include <iostream>

//Functions:
size_t GetIterationNumber(
    const double& inlierRatio_,
    const double& confidence_,
    const size_t& sampleSize_)
{
    std::cout << "Inlier ratio is " << inlierRatio_ << std::endl;
    double a =
        log(1.0 - confidence_);
    double b =
        log(1.0 - std::pow(inlierRatio_, sampleSize_));

    if (abs(b) < std::numeric_limits<double>::epsilon())
        return std::numeric_limits<size_t>::max();

    return a / b;
}

std::vector<int> RandomPerm(
    const int& sampleSize,
    const int& dataSize) {
    std::vector<int> result;
    for (size_t sampleIdx = 0; sampleIdx < sampleSize && sampleIdx < dataSize; sampleIdx++) {
        int val;
        bool isFound;
        do {
            val = rand() % (dataSize - 1);
            isFound = false;
            for (size_t i = 0; i < result.size(); i++) {
                if (val == result[i]) {
                    isFound = true;
                    break;
                }
            }
        } while (isFound);
        result.push_back(val);
    }
    return result;
}

void FitModel(
    const std::vector<cv::Point3d>& points,
    std::vector<int>& inliersIdx,
    const Plane& planeModel,
    const double& threshold) {
    cv::Point3d point;
    double distance;
    //Fit model:
    inliersIdx.clear();
    for (size_t dataIdx = 0; dataIdx < points.size(); dataIdx++) {
        point = points[dataIdx];
        distance = abs(planeModel.a * point.x + planeModel.b * point.y + planeModel.c * point.z + planeModel.d);
        if (distance < threshold) {
            inliersIdx.push_back(dataIdx);
        }
    }
}

void FitPlaneLSQ(const std::vector<cv::Point3d>& points,
    const std::vector<int>& inliers,
    Plane& planeModel) {
    //returning planeModel
    std::vector<cv::Point3d> normalizedPoints;

    cv::Point3d masspoint(0, 0, 0);

    for (const auto& inlierIdx : inliers)
    {
        masspoint += points[inlierIdx];
        normalizedPoints.push_back(points[inlierIdx]);
    }
    masspoint = masspoint * (1.0 / inliers.size());

    // Move the point cloud to have the origin in their mass point
    for (auto& point : normalizedPoints)
        point -= masspoint;

    // Calculating the average distance from the origin
    double averageDistance = 0.0;
    for (auto& point : normalizedPoints)
    {
        averageDistance += cv::norm(point);
    }

    averageDistance /= normalizedPoints.size();
    const double ratio = sqrt(3) / averageDistance;

    // Making the average distance to be sqrt(2)
    for (auto& point : normalizedPoints)
        point *= ratio;

    // Now, we should solve the equation.
    cv::Mat A(normalizedPoints.size(), 3, CV_64F); // ??

    // Building the coefficient matrix
    for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
    {
        const size_t& rowIdx = pointIdx;

        A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
        A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
        A.at<double>(rowIdx, 2) = normalizedPoints[pointIdx].z;
    }

    cv::Mat evals, evecs;
    cv::eigen(A.t() * A, evals, evecs);

    const cv::Mat& normal = evecs.row(2);
    const double a = normal.at<double>(0);
    const double b = normal.at<double>(1);
    const double c = normal.at<double>(2);
    const double d = -(a * masspoint.x + b * masspoint.y + c * masspoint.z);

    planeModel.a = a;
    planeModel.b = b;
    planeModel.c = c;
    planeModel.d = d;
}

void LORANSAC_LSQ(const std::vector<cv::Point3d>& dataset,
    std::vector<int>& bestInliersIdx,
    Plane& bestModel,
    const int kSampleSize,
    double threshold,
    double confidence,
    int maxIter) {

    if (dataset.size() < kSampleSize) {
        std::cout << "Not enough data for RANSAC." << std::endl;
        return;
    }

    std::cout << "RANSAC is running." << std::endl;
    size_t maxIter_ = maxIter;

    size_t iter = 0;
    size_t bestInlierNumber = 0;

    std::vector<int> sampleIdx(kSampleSize);
    std::vector<cv::Point3d> sample(kSampleSize);

    std::vector<int> inliersIdx;

    cv::Mat tmpImg;

    while (iter++ < maxIter_) {

        //Select random points
        sample.clear();
        sampleIdx = RandomPerm(kSampleSize, dataset.size());
        for (size_t i = 0; i < kSampleSize; i++) {
            sample.push_back(dataset[sampleIdx[i]]);
        }

        //Create model:
        cv::Point3d A = sample[0];
        cv::Point3d B = sample[1];
        cv::Point3d C = sample[2];
        cv::Point3d AC = C - A;
        cv::Point3d AB = B - A;

        cv::Point3d n = AC.cross(AB);

        n = n / cv::norm(n);

        double a = n.x;
        double b = n.y;
        double c = n.z;
        double d = -(a * A.x + b * A.y + c * A.z);
        Plane plane;
        plane.a = a;
        plane.b = b;
        plane.c = c;
        plane.d = d;
        cv::Point3d point;
        double distance;
        //Fit model:
        FitModel(dataset, inliersIdx, plane, threshold);

        //Check the model:
        if (inliersIdx.size() > bestInlierNumber) {
            bestModel = plane;
            bestInliersIdx = inliersIdx;

            //Local optimization:
            Plane model_LO;
            std::vector<int> inliersIdx_LO;
            FitPlaneLSQ(dataset, inliersIdx, model_LO); 
            FitModel(dataset, inliersIdx_LO, model_LO, threshold);
            if (bestInliersIdx.size() < inliersIdx_LO.size()) {
                bestInliersIdx = inliersIdx_LO;
                bestModel = model_LO;
            }
            bestInlierNumber = bestInliersIdx.size();
            //End of local optimization:

            // Update the maximum iteration number
            maxIter_ = GetIterationNumber(
                static_cast<double>(bestInlierNumber) / static_cast<double>(dataset.size()),
                confidence,
                kSampleSize);

            printf("Inlier number = %d\tMax iterations = %d\n", (int)bestInliersIdx.size(), (int)maxIter_);
        }

        std::cout << (double)iter / maxIter_ * 100 << "%\r";
        std::cout.flush();
    }
}
