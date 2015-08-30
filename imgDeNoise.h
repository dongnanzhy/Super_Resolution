#pragma once
#include "opencv2\opencv.hpp"
using namespace cv;

class imgDeNoise
{
public:
	imgDeNoise(void);
	~imgDeNoise(void);
	Mat wavelet_denoise(Mat test_img, float SHRINKAGE_T);
	Mat wavelet_denoise_HF(Mat test_img, float SHRINKAGE_T);
	void cvHaarWavelet(Mat &src,Mat &dst,int NIter);
	void cvInvHaarWavelet(Mat &src,Mat &dst,int NIter, int SHRINKAGE_TYPE=0, float SHRINKAGE_T=50);

private:
	float sgn(float x);
	float soft_shrink(float d,float T);
	float hard_shrink(float d,float T);
	float Garrot_shrink(float d,float T);
};

