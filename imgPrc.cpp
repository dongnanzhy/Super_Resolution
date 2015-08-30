//============================================================ 
/*                                                          */
/*      TYPE           : C++ Code                           */
/*      FUNCTION       :                                    */
/*      Author         : Yan Zhao                           */
/*      Rev,DATE       : 1.0,05 Aug, 2015					*/
/*                                                          */
//============================================================
#include <iostream>
#include "opencv2\opencv.hpp"
#include "imgPrc.h"

using namespace cv;

imgPrc::imgPrc(void)
{
}

imgPrc::~imgPrc(void)
{
}

//--------------------------------
// resize image, downsize using area interpolation, upsize using cubic interpolation
//--------------------------------
Mat imgPrc::zoom(Mat image, double scale)
{
	Mat dst;//dst image
	if (scale > 1)
	{
        resize(image, dst, Size(), scale, scale, INTER_CUBIC);  // a bicubic interpolation over 4x4 pixel neighborhood
	} else
	{
		resize(image, dst, Size(), scale, scale, INTER_AREA);  // resampling using pixel area relation.
	}
	return dst;
}
//--------------------------------
// ITU-R BT.709 conversion
//--------------------------------
Mat imgPrc::rgb2Ycc(Mat rgb_image)
{
	Mat dst = rgb_image.clone();
	Mat trans = (Mat_<double>(3,3) << 0.0722, 0.7152, 0.2126, 0.5, -0.3854, -0.1146, -0.0458, -0.4542, 0.5);
	
	for (int i = 0; i < dst.size().height; i++)
	{
		for (int j = 0; j < dst.size().width; j++)
		{
			Mat pixel(Vec3d(dst.at<Vec3s>(i,j)),false);
			Mat tmp = trans * pixel;
			dst.at<Vec3s>(i,j) = Vec3s(tmp);
		}
	}
	return dst;
}
//--------------------------------
// ITU-R BT.601 conversion, opencv
//--------------------------------
Mat imgPrc::rgb2Ycc_cv(Mat rgb_image)
{
	Mat dst;
	cvtColor(rgb_image,dst,CV_BGR2YCrCb,0);
	return dst;
}
//--------------------------------
// ITU-R BT.709 conversion
//--------------------------------
Mat imgPrc::ycc2Rgb(Mat ycc_image)
{
	Mat dst = ycc_image.clone();
	Mat trans = (Mat_<double>(3,3) << 1, 1.8556, 0, 1, -0.1873, -0.4681, 1, 0, 1.5748);
	
	for (int i = 0; i < dst.size().height; i++)
	{
		for (int j = 0; j < dst.size().width; j++)
		{
			Mat pixel(Vec3d(dst.at<Vec3s>(i,j)),false);
			Mat tmp = trans * pixel;
			dst.at<Vec3s>(i,j) = Vec3s(tmp);
		}
	}
	return dst;
}
//--------------------------------
// ITU-R BT.601 conversion, opencv
//--------------------------------
Mat imgPrc::ycc2Rgb_cv(Mat ycc_image)
{
	Mat dst;
	cvtColor(ycc_image,dst,CV_YCrCb2BGR,0);
	return dst;
}
//--------------------------------
// split Y Channel from ycc image
//--------------------------------
Mat imgPrc::getYChannel(Mat ycc_image)
{
   vector<Mat> channels(3);
   split(ycc_image, channels);
   return channels[0];
}
//--------------------------------
// split color Channel from ycc image
//--------------------------------
vector<Mat> imgPrc::getCbCrChannel(Mat ycc_image)
{
   vector<Mat> dst(2);
   vector<Mat> channels(3);
   split(ycc_image, channels);
   dst[0] = channels[1];
   dst[1] = channels[2];
   return dst;
}
//--------------------------------
// input: m1--reference image, m2--evaluated image
// output: PSNR double value
// calculate PSNR by sum of RGB channels
//--------------------------------
double imgPrc::getPSNR(Mat m1, Mat m2)
{
	Mat s1;
    absdiff(m1, m2, s1);       // |M1 - M2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |M1 - M2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /double(m1.channels() * m1.total());
        double psnr = 10.0*log10((1023*1023)/mse);
        return psnr;
    }
}
//--------------------------------
// input: m1--reference image, m2--evaluated image
// output: PSNR double value
// calculate PSNR by Y Channel
//--------------------------------
double imgPrc::getPSNR_Y(Mat m1, Mat m2)
{
	Mat ycc_m1, ycc_m2;
	cvtColor(m1,ycc_m1,CV_BGR2YCrCb,0);
	cvtColor(m2,ycc_m2,CV_BGR2YCrCb,0);

	Mat s1;
    absdiff(ycc_m1, ycc_m2, s1);       // |M1 - M2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |M1 - M2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0];

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /double(ycc_m1.total());
        double psnr = 10.0*log10((1023*1023)/mse);
        return psnr;
    }
}
//--------------------------------
// input: m1--reference image, m2--evaluated image
// output: MSE double value
// calculate MSE
//--------------------------------
double imgPrc::getMSE(Mat m1, Mat m2)
{
	Mat s1;
    absdiff(m1, m2, s1);       // |M1 - M2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |M1 - M2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        
        return sse;
    }
}
//--------------------------------
// input: m1--reference image, m2--evaluated image
// output: Scalar of SSIM on threee channels(RGB)
// calculate SSIM by RGB Channels
//--------------------------------
Scalar imgPrc::getSSIM(Mat m1, Mat m2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    m1.convertTo(I1, d);           // cannot calculate on one byte large values
    m2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}
//--------------------------------
// input: image of RGB Channel
// output: noised image of RGB Channel
// add gaussian noise manually, for wavelet denoise
//--------------------------------
Mat imgPrc::addGaussianNoise(Mat image)
{
	int type_tmp = image.type();
	image.convertTo(image,CV_32FC3);
	double m = 0;
	double M = 0;
	minMaxLoc(image,&m,&M);
	Mat noise = Mat(image.size(),image.type());
	Mat tmp_mat = Mat(image.size(),image.type());
	Mat dst =  Mat(image.size(),image.type());

	normalize(image, tmp_mat, 0.0, 1.0, CV_MINMAX, -1);
	randn(noise, 0, 0.03);
	tmp_mat = tmp_mat + noise;
	normalize(tmp_mat, dst, 0.0, 1.0, CV_MINMAX, -1);
	dst = dst * (M-m) + m;
	dst.convertTo(dst, type_tmp);
	return dst;
}
//--------------------------------
// input: image of RGB Channel
// output: detail enhanced image of RGB Channel
// enhance details using USM method, with sigma=1.4 gaussian blur
//--------------------------------
Mat imgPrc::unsharpMask(Mat image)
{
	Mat tmp = Mat(image.size(),image.type());
	Mat out = Mat(image.size(),image.type());
    GaussianBlur(image, tmp, Size(7,7), 1.4);
    addWeighted(image, 2, tmp, -1, 0,out);
	return out;
}
//--------------------------------
// input: HF components--1 channel
// output: Hist stretched HF components
// manually find most frequently predicted value, set to 0, and strech other values------> for error analysis
//--------------------------------
Mat imgPrc::stretch(Mat HF_tmp)
{
	// (manually find most freuntly occured value)
	vector<int> counts(50, 0);
	for (auto i = 0; i < HF_tmp.rows; i++)
	{
		for (auto j = 0; j < HF_tmp.cols; j++)
		{
			if (HF_tmp.at<short int>(i,j) >= 50 || HF_tmp.at<short int>(i,j) < 0) continue;
			counts.at(HF_tmp.at<short int>(i,j)) += 1;
		}
	}
	int threshold = distance(counts.begin(), max_element(counts.begin(),counts.end()));
	// (stretch histogram)
	Mat rst = Mat(HF_tmp.size(),HF_tmp.type());
	for (auto i = 0; i < rst.rows; i++)
	{
		for (auto j = 0; j < rst.cols; j++)
		{
			short int tmp = HF_tmp.at<short int>(i,j);
			if (tmp < threshold) {
				rst.at<short int>(i,j) = tmp * 2;
			} else if (tmp == threshold ) {
				rst.at<short int>(i,j) = 0;
			} else {
				rst.at<short int>(i,j) = tmp / 5 ;
			}
		}
	}
	return rst;
}