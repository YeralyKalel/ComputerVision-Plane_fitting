#pragma once

#include <opencv2/core.hpp>
#include "Plane.h"

size_t GetIterationNumber(
    const double& inlierRatio_,
    const double& confidence_,
    const size_t& sampleSize_);

std::vector<int> RandomPerm(
    const int& sampleSize,
    const int& dataSize);

void FitModel(
    const std::vector<cv::Point3d>& points,
    std::vector<int>& inliersIdx,
    const Plane& planeModel,
    const double& threshold);

void FitPlaneLSQ(
    const std::vector<cv::Point3d>& points,
    const std::vector<int>& inliers,
    Plane& planeModel);

void LORANSAC_LSQ(
    const std::vector<cv::Point3d>& dataset,
    std::vector<int>& bestInliersIdx,
    Plane& bestModel,
    const int sampleSize,
    double threshold,
    double confidence,
    int maxIter);