#pragma once

using namespace cv;

class imgIO
{
public:
	imgIO(void);
	~imgIO(void);
	Mat imgRead(string filePath);
	void imgDisp(Mat image);
	void imgWrite(Mat image, string outPath);
	void xmlWrite(Mat m, char* filePath);
	Mat xmlRead( char* filePath);
};

