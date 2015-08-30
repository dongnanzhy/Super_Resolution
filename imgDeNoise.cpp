//============================================================ 
/*                                                          */
/*      TYPE           : C++ Code                           */
/*      FUNCTION       :                                    */
/*      Author         : Yan Zhao                           */
/*      Rev,DATE       : 1.0,05 Aug, 2015					*/
/*                                                          */
//============================================================
#include "imgDeNoise.h"
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace std;

// Filter type
#define NONE 0  // no filter
#define HARD 1  // hard shrinkage
#define SOFT 2  // soft shrinkage
#define GARROT 3  // garrot filter

imgDeNoise::imgDeNoise(void)
{
}


imgDeNoise::~imgDeNoise(void)
{
}

//--------------------------------
// Wavelet denoise for noised image
//--------------------------------
Mat imgDeNoise::wavelet_denoise(Mat test_img, float SHRINKAGE_T)
{
	const int NIter=4;
	int cols = test_img.cols;
	int rows = test_img.rows;
	Mat ycc_image;
    cvtColor(test_img,ycc_image,CV_BGR2YCrCb,0);
	vector<Mat> channels(3);
    split(ycc_image, channels);

	Mat Src;
	channels[0].convertTo(Src,CV_32FC1);
	Mat Dst=Mat(rows, cols, CV_32FC1);
    Mat Temp=Mat(rows, cols, CV_32FC1);
    Mat Filtered=Mat(rows, cols, CV_32FC1);
	cvHaarWavelet(Src,Dst,NIter);
	Dst.copyTo(Temp);
	cvInvHaarWavelet(Temp,Filtered,NIter,GARROT,SHRINKAGE_T);
	Filtered.convertTo(Filtered,channels[1].type());
	// image size cannot be divided 
	Filtered(Rect(0, 0, cols-cols%(1<<NIter), rows-rows%(1<<NIter))).copyTo(channels[0](Rect(0, 0, cols-cols%(1<<NIter), rows-rows%(1<<NIter))));
	//merge channels back
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	//restore rgb image
	Mat out_rgb;
	cvtColor(out_ycc,out_rgb,CV_YCrCb2BGR);
    return out_rgb;
}
//--------------------------------
// Wavelet denoise for HF components
//--------------------------------
Mat imgDeNoise::wavelet_denoise_HF(Mat test_img, float SHRINKAGE_T)
{
	const int NIter=4;
	int cols = test_img.cols;
	int rows = test_img.rows;

	Mat Src;
	test_img.convertTo(Src,CV_32FC1);
	Mat Dst=Mat(rows, cols, CV_32FC1);
    Mat Temp=Mat(rows, cols, CV_32FC1);
    Mat Filtered=Mat(rows, cols, CV_32FC1);
	cvHaarWavelet(Src,Dst,NIter);
	Dst.copyTo(Temp);
	cvInvHaarWavelet(Temp,Filtered,NIter,GARROT,SHRINKAGE_T);
	// image size cannot be divided 
	Filtered(Rect(0, 0, cols-cols%(1<<NIter), rows-rows%(1<<NIter))).copyTo(test_img(Rect(0, 0, cols-cols%(1<<NIter), rows-rows%(1<<NIter))));
    return test_img;
}

//--------------------------------
// Wavelet transform
//--------------------------------
void imgDeNoise::cvHaarWavelet(Mat &src, Mat &dst, int NIter)
{
	float c,dh,dv,dd;
    assert( src.type() == CV_32FC1 );
    assert( dst.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    for (int k=0;k<NIter;k++) 
    {
        for (int y=0;y<(height>>(k+1));y++)
        {
            for (int x=0; x<(width>>(k+1));x++)
            {
                c= float((src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1)) * 0.5);
                dst.at<float>(y,x)=c;

                dh=float((src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1)) * 0.5);
                dst.at<float>(y,x+(width>>(k+1)))=dh;

                dv=float((src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1)) * 0.5);
                dst.at<float>(y+(height>>(k+1)),x)=dv;

                dd=float((src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1)) * 0.5);
                dst.at<float>(y+(height>>(k+1)),x+(width>>(k+1)))=dd;
            }
        }
        dst.copyTo(src);
    }  
}
//--------------------------------
//Inverse wavelet transform
//--------------------------------
void imgDeNoise::cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE, float SHRINKAGE_T)
{   
	float c,dh,dv,dd;
    assert( src.type() == CV_32FC1 );
    assert( dst.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    //--------------------------------
    // NIter - number of iterations 
    //--------------------------------
    for (int k=NIter;k>0;k--) 
    {
        for (int y=0;y<(height>>k);y++)
        {
            for (int x=0; x<(width>>k);x++)
            {
                c=src.at<float>(y,x);
                dh=src.at<float>(y,x+(width>>k));
                dv=src.at<float>(y+(height>>k),x);
                dd=src.at<float>(y+(height>>k),x+(width>>k));

               // (shrinkage)
                switch(SHRINKAGE_TYPE)
                {
                case HARD:
                    dh=hard_shrink(dh,SHRINKAGE_T);
                    dv=hard_shrink(dv,SHRINKAGE_T);
                    dd=hard_shrink(dd,SHRINKAGE_T);
                    break;
                case SOFT:
                    dh=soft_shrink(dh,SHRINKAGE_T);
                    dv=soft_shrink(dv,SHRINKAGE_T);
                    dd=soft_shrink(dd,SHRINKAGE_T);
                    break;
                case GARROT:
                    dh=Garrot_shrink(dh,SHRINKAGE_T);
                    dv=Garrot_shrink(dv,SHRINKAGE_T);
                    dd=Garrot_shrink(dd,SHRINKAGE_T);
                    break;
                }

                //-------------------
                dst.at<float>(y*2,x*2)=float (0.5*(c+dh+dv+dd));
                dst.at<float>(y*2,x*2+1)=float (0.5*(c-dh+dv-dd));
                dst.at<float>(y*2+1,x*2)=float (0.5*(c+dh-dv-dd));
                dst.at<float>(y*2+1,x*2+1)=float (0.5*(c-dh-dv+dd));            
            }
        }
        Mat C=src(Rect(0,0,width>>(k-1),height>>(k-1)));
        Mat D=dst(Rect(0,0,width>>(k-1),height>>(k-1)));
        D.copyTo(C);
    }   
}

//--------------------------------
// signum
//--------------------------------
float imgDeNoise::sgn(float x)
{
	float res=0;
    if(x==0)
    {
        res=0;
    }
    if(x>0)
    {
        res=1;
    }
    if(x<0)
    {
        res=-1;
    }
    return res;
}
//--------------------------------
// Soft shrinkage
//--------------------------------
float imgDeNoise::soft_shrink(float d, float T)
{
	float res;
    if(T<fabs(d))
    {
        res=sgn(d)*(fabs(d)-T);
    }
    else
    {
        res=0;
    }

    return res;

}
//--------------------------------
// Hard shrinkage
//--------------------------------
float imgDeNoise::hard_shrink(float d, float T)
{
	float res;
    if(fabs(d)>T)
    {
        res=d;
    }
    else
    {
        res=0;
    }

    return res;
}
//--------------------------------
// Garrot shrinkage
//--------------------------------
float imgDeNoise::Garrot_shrink(float d, float T)
{
	float res;
    if(fabs(d)>T)
    {
        res=d-((T*T)/d);
    }
    else
    {
        res=0;
    }

    return res;
}