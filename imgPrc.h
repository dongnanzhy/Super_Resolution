#pragma once
using namespace cv;


class imgPrc
{
public:
	imgPrc(void);
	~imgPrc(void);
	Mat zoom(Mat image, double scale);
	Mat rgb2Ycc(Mat rgb_image);
    Mat rgb2Ycc_cv(Mat rgb_image);
	Mat ycc2Rgb(Mat ycc_image);
	Mat ycc2Rgb_cv(Mat ycc_image);
	Mat getYChannel(Mat ycc_image);
	vector<Mat> getCbCrChannel(Mat ycc_image);
	double getPSNR(Mat m1, Mat m2);
	double getPSNR_Y(Mat m1, Mat m2);
    double getMSE(Mat m1, Mat m2);
	Scalar getSSIM(Mat m1, Mat m2);
	Mat addGaussianNoise(Mat image);
	Mat unsharpMask(Mat image);
	Mat stretch(Mat HF);
};

