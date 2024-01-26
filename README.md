# Fingerprint-Recgonition Minutiae Algorithm and CNN

This project uses an hybrid solution to utilise the best of both the algorithms: Minutiae Algorithm and Convolutional Neural Network(CNN)

The FVC2002 dataset was used for this project as it was found to be the best dataset for Fingerprint Applications.

The minutiae points of each fingerpirnt is extracted using the Minutiae algorithm. Using these points, an image grid of 0s is formed, where only the minutiae points are 1s.

Each fingerprint is represented as this minutiae grid and flattened before feeding it to the CNN for training and prediction.