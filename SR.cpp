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
#include "imgPrc.h"
#include "ml_nn.h"
#include "imgDeNoise.h"
using namespace std;
using namespace cv;


imgIO *p_img_io;
imgPrc *p_img_prc;
ml_nn *p_ann;
imgDeNoise *p_img_deNoise;

int main(int argc, char** argv) 
{
   p_img_io = new imgIO();
   p_img_prc = new imgPrc();
   p_ann = new ml_nn();

   if( argc < 2)
   {
      cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
      return -1;
   }

   /*
      ********************************
      Save Training Data
	  ********************************
   */
   /*
   vector<Mat> images(6);
   vector<string> paths(3);
   paths[0] = "m1-1\\m1-1_";
   paths[1] = "m2-1\\m2-1_";
   paths[2] = "m6-3\\m6-3_";
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\";
   for (size_t  i = 0; i < paths.size(); i++)
   {
	   for (int j = 0; j < 2; j++)
	   {
		   string path = source + paths[i] + format("%04d", j*5) + ".ppm";
		   Mat image = p_img_io->imgRead(path);
		   Mat image_tmp = p_img_prc->zoom(image,0.25);
		   images[i*2+j] = image_tmp(Rect(image_tmp.cols/4, 0, image_tmp.cols/2,image_tmp.rows)).clone();
	   }
   }
   	for (size_t  k = 0; k < images.size();k++)
	{
		String path_origin =  format("%04d", k)+"_HR.ppm";
		String path_LR =  format("%04d", k)+"_LR.ppm";
		String path_HF =  format("%04d", k)+"_HF.pgm";
		Mat LR;
		GaussianBlur( images[k], LR, Size(7,7), 1.4, 1.4);
	    vector<Mat> channel_vec = p_ann->get_LRHF(images[k]);
		Mat HF;
		channel_vec[1].convertTo(HF,CV_16SC1);
		p_img_io->imgWrite(LR,path_LR);
		p_img_io->imgWrite(images[k],path_origin);
		p_img_io->imgWrite(channel_vec[1],path_HF);
	}
   Mat trainData = p_ann->get_train_normPatch(images,5, false);
   p_img_io->xmlWrite(trainData,"trainData.xml");
   */


   /*
     ********************************
      Save Training Data with Noise
     ********************************
   */
   /*
   vector<Mat> images(6);
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\Noised_Data\\";
   for (size_t  i = 0; i < 6; i++)
   {
		string path = source + format("%04d", i) + "_HR.ppm";
		Mat image = p_img_io->imgRead(path);
		images[i] = image/64;
   }
   Mat trainData = p_ann->get_train_normPatch(images,5, true);
   p_img_io->xmlWrite(trainData,"trainData.xml");
   */

   /* 
      ********************************
      Get Test Data
	  ********************************
   */
   /*
   string paths = "m11-1\\m11-1_";
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\";
   string path = source + paths + format("%04d", 0) + ".ppm";
   Mat image = p_img_io->imgRead(path);
   Mat image_tmp = p_img_prc->zoom(image,0.25);
   Mat image_exp = image_tmp(Rect(image_tmp.cols/4, 0, image_tmp.cols/2,image_tmp.rows)).clone();
   Mat LR = Mat(image_exp.size(), image_exp.type());
   GaussianBlur(image_exp, LR, Size(7,7), 1.4, 1.4);
   p_img_io->xmlWrite(image_exp,"original.xml");
   p_img_io->xmlWrite(LR,"LR_test.xml");

   Mat ycc_img = p_img_prc->rgb2Ycc_cv(image_exp);
   Mat ycc_LR = p_img_prc->rgb2Ycc_cv(LR);
   // get YChannel
   Mat yChannel_img = p_img_prc->getYChannel(ycc_img);
   Mat yChannel_LR = p_img_prc->getYChannel(ycc_LR);
   // compute HF
   Mat yChannel_HF;
   subtract(yChannel_img, yChannel_LR, yChannel_HF,noArray(),CV_16SC1);
   p_img_io->xmlWrite(yChannel_HF,"orig_HF.xml");
   
   p_img_io->imgWrite(image_exp, "origin.ppm");
   p_img_io->imgWrite(yChannel_HF,"orig_HF.pgm");
   p_img_io->imgWrite(LR, "LR_img.ppm");
   */

   /*
      ********************************
      Cross Validation
	  ********************************
   */  
   //CvSVM* model_svm = p_ann->model_read_svm("SVM_model.xml");
   //Mat trainData = p_img_io->xmlRead("trainData.xml");
   //Mat cv_testData = trainData(Rect(0, 0, trainData.cols -1, trainData.rows)).clone();
   //p_img_io->xmlWrite(trainData.col(trainData.cols-1),"input_label.xml");
   //Mat cv_test_result = p_ann->perform_cv(cv_testData, model_svm);
   //p_img_io->xmlWrite(cv_test_result,"output_label.xml");


   /*
      ********************************
      Train and save model
	  ********************************
   */
   //Mat trainData = p_img_io->xmlRead("trainData.xml"); 
   //cout<< "data read, 1000th row is" << trainData.row(1000) << endl; 
   //CvANN_MLP* nnetwork = p_ann->nn_model(trainData);
   //CvSVM* model_svm = p_ann->svm_model(trainData);
   //CvKNearest* model_KNN = p_ann->kNN_model(trainData);
   //p_ann->model_write(nnetwork,"ANN_model_30.xml");
   //p_ann->model_write(model_svm,"SVM_model.xml");
   //p_ann->model_write(model_KNN,"KNN_model.xml");
   

   /*
      ********************************
      Perform & Evaluate model
	  ********************************
   */
   /*
   string paths = "m11-1\\m11-1_";
   string path = argv[1] + paths + format("%04d", 0) + ".ppm";
   Mat image = p_img_io->imgRead(path);
   Mat image_tmp = p_img_prc->zoom(image,0.25);
   Mat image_exp = image_tmp(Rect(image_tmp.cols/4, 0, image_tmp.cols/2,image_tmp.rows)).clone();
   Mat LR = Mat(image_exp.size(), image_exp.type());
   GaussianBlur(image_exp, LR, Size(7,7), 1.4, 1.4);

   CvANN_MLP* nnetwork = p_ann->model_read_nn("ANN_model_90.xml");
   //CvSVM* model_svm = p_ann->model_read_svm("SVM_model.xml");
   Mat output = p_ann->perform_normPatch(LR, nnetwork, 5);
   //p_img_io->xmlWrite(output,"Final_test.xml");
   Mat test_img = LR;
   Mat restored = output;
   */

   /*
      ********************************
      Perform model on 4K
	  ********************************
   */

   string paths = "m2-4-1\\m2-4-1_";
   string path = argv[1] + paths + format("%04d", 0) + ".ppm";
   Mat LR = p_img_io->imgRead(path);
   CvANN_MLP* nnetwork = p_ann->model_read_nn("ANN_model.xml");
   Mat restored = p_ann->perform_4K(LR, nnetwork, 5);
   //Mat HF = p_img_io->xmlRead("restored HF.xml");
   //Mat restored = p_ann->restore_from_hf(LR,HF);
   

   /*
      ********************************
      Perform & Evaluate noised data
	  ********************************
   */
	/*
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\Noised_Data\\";
   string path = source + "LR_img.ppm";
   Mat image = p_img_io->imgRead(path);
   Mat LR  = image/64;
   CvANN_MLP* nnetwork = p_ann->model_read_nn("ANN_model_70.xml");
   Mat output = p_ann->perform_normPatch(LR, nnetwork, 5);
   path = source + "HR_img.ppm";
   Mat image_exp = p_img_io->imgRead(path);
   image_exp = image_exp / 64;
   Mat test_img = LR;
   Mat restored = output;
   */

    /*
      ********************************
      Test unsharpen mask
	  ********************************
   */
	
   //string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\Noised_Data\\";
   //string path = source + "LR_img.ppm";
   //Mat LR = p_img_io->imgRead(path);
   //LR  = LR/64;
   //string paths = "m2-4-1\\m2-4-1_";
   //string path = argv[1] + paths + format("%04d", 0) + ".ppm";
   //Mat LR = p_img_io->imgRead(path);
   //Mat restored = p_img_prc->unsharpMask(LR);
   //path = source + "HR_img.ppm";
   //Mat image_exp = p_img_io->imgRead(path);
   //image_exp = image_exp / 64;
   //Mat test_img = LR;
   

   /*
      ********************************
      Compute reference ppm
	  ********************************
   */
   /*
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\Noised_Data\\";
   string path = source + "Ref_2.ppm";
   Mat restored = p_img_io->imgRead(path);
   restored = restored / 64;
   path = source + "HR_img.ppm";
   Mat image_exp = p_img_io->imgRead(path);
   image_exp = image_exp / 64;
   path = source + "LR_img.ppm";
   Mat test_img = p_img_io->imgRead(path);
   test_img = test_img / 64;
   */

   /*
      ********************************
      Denoise
	  ********************************
   */  
   //Mat image_exp = p_img_io->xmlRead("original.xml");
   //Mat test_img = p_img_prc->addGaussianNoise(image_exp);
   //Mat restored = p_img_deNoise->wavelet_denoise(test_img,20);
   //Mat test_img = p_img_io->xmlRead("LR_test.xml");
   //p_img_io->xmlWrite(denoised_img,"Denoised.xml");


   /*
      ********************************
      Compute PSNR and SSIM
	  ********************************
   */
   /*
   //Mat image_exp = p_img_io->xmlRead("original.xml");
   // (compute MSE)
   double mse = p_img_prc->getMSE(test_img,image_exp);
   double mse_restored = p_img_prc->getMSE(image_exp,restored);
   cout << "mse before= " << endl << mse << endl;
   cout << "mse after= " << endl << mse_restored << endl;
   // (compute PSNR)
   double psnr = p_img_prc->getPSNR(image_exp, test_img);
   double psnr_restored = p_img_prc->getPSNR(image_exp,restored);
   cout << "psnr before= " << endl << psnr << endl;
   cout << "psnr after= " << endl << psnr_restored << endl;
   // (compute PSNR)
   double psnr_Y = p_img_prc->getPSNR_Y(image_exp, test_img);
   double psnr_Y__restored = p_img_prc->getPSNR_Y(image_exp,restored);
   cout << "Y channel psnr before= " << endl << psnr_Y << endl;
   cout << "Y channel psnr after= " << endl << psnr_Y__restored << endl;
   // (compute SSIM)
   Scalar ssim_before = p_img_prc->getSSIM(test_img,image_exp);
   Scalar ssim_after = p_img_prc->getSSIM(restored,image_exp);
   cout << "ssim before= " << endl << ssim_before << endl;
   cout << "ssim after= " << endl << ssim_after << endl;
  */


   /*
      ********************************
      Disp and store Img
	  ********************************
   */
   Mat HF = p_img_io->xmlRead("restored HF.xml");
   HF.convertTo(HF, CV_16SC1);
   p_img_io->imgWrite(HF,"HF.pgm");
   p_img_io->imgWrite(restored, "restored.ppm");


   system("pause");
   return 0;
}