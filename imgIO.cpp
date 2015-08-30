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
#include "imgIO.h"


using namespace cv;
using namespace std;

imgIO::imgIO(void)
{
}

imgIO::~imgIO(void)
{
}
//--------------------------------
// read image from file path
// given PPM image are 10 bis, read as CV_16UC3
//--------------------------------
Mat imgIO::imgRead(string filePath)
{
	 Mat image = imread(filePath,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR); // Read the file
	 if( !image.data ) // Check for invalid input
     {
      cout << "Could not open or find the image" << endl ;
	  system ("pause");
     }
	 return image;
}
//--------------------------------
// display image
//--------------------------------
void imgIO::imgDisp(Mat image)
{
	cout << image.size().height<<","<<  image.size().width << endl;
	cout << image.dims<<endl;       // 2 dimension
	cout << image.depth()<<endl;    // depth per channel  CV_16U
	// - when display opencv will automatically map 16 bits to 8 bits
	// - so map from 10 bits to 16 bis by multiplying 2^6,
	Mat dst = image * 64;   
	namedWindow( "Display window" ); // Create a window for display.
    imshow( "Display window", dst ); // Show our image inside it.
}
//--------------------------------
// write image to ppm format
//--------------------------------
void imgIO::imgWrite(Mat image, string outPath)
{
	Mat dst;
	// - for rgb image, map from 10 bits to 16 bits
	if (image.channels() > 1)
	{
		dst = image * 64;
	} 
	// - for HF components, shift 0 value to 127
	else
	{
		dst=image+1024;
        dst = (dst/8 -128) * 10 + 128;
        dst.convertTo(dst,CV_8UC1);
	}
	vector<int> compression_params; // Stores the compression parameters
    compression_params.push_back(CV_IMWRITE_PXM_BINARY); // Set to PXM compression
    compression_params.push_back(1); 
    string imageFilename = outPath; // Some file name - C++ requires an std::string
    imwrite(imageFilename, dst, compression_params); // Write matrix to file
}
//--------------------------------
// write image matrix to xml file
//--------------------------------
void imgIO::xmlWrite(Mat m, char* filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);
	fs << "Matrix" << m;
	fs.release();
}
//--------------------------------
// read image matrix from xml file
//--------------------------------
Mat imgIO::xmlRead(char* filePath)
{
	FileStorage fs(filePath, FileStorage::READ);
	Mat dst;
	fs["Matrix"] >> dst;
	fs.release();
	return dst;
}