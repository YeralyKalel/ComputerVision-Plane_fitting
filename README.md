# ComputerVision-Plane_fitting

Concatenates two images to create panorama view (can be extended into multiple images)

1) Load the images and matrix containing corresponding points
2) Estimate Homography using LO-RANSAC
3) Concatenate the images by estimated common homography
4) Print the panoramic image
