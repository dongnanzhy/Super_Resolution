#pragma once
#include "imgPrc.h"
using namespace cv;

class ml_nn
{
public:
	ml_nn(void);
	~ml_nn(void);
	int close_ml_nn(void);
	Mat get_train(vector<Mat> images,int patch_size);
	Mat get_train_normPatch(vector<Mat> images,int patch_size, bool isNoised);
	Mat get_CV_test(Mat train_data);
	CvANN_MLP* nn_model(Mat train_data);
	CvSVM* svm_model(Mat train_data);
	CvKNearest* kNN_model(Mat train_data);
	//void construct_cnn();
	Mat perform_cv(Mat test_data, CvANN_MLP* nnetwork);
	Mat perform_cv(Mat test_data, CvSVM* model_svm);
    Mat perform(Mat test_img, CvANN_MLP* nnetwork, int patch_size);
	Mat perform_normPatch(Mat test_img, CvANN_MLP* nnetwork, int patch_size);
	Mat perform_4K(Mat test_img, CvANN_MLP* nnetwork, int patch_size);
	Mat perform(Mat test_img, CvSVM* model_svm, int patch_size);
	Mat perform(Mat test_img, CvKNearest* model_KNN, int patch_size);
	void model_write(CvANN_MLP* nnetwork, char* filePath);
	void model_write(CvSVM* model_svm, char* filePath);
	void model_write(CvKNearest* model_KNN, char* filePath);
	CvANN_MLP* model_read_nn( char* filePath);
	CvSVM* model_read_svm(char* filePath);
    CvKNearest* model_read_KNN(char* filePath);
	Mat restore_from_hf(Mat test_img, Mat HF);
    vector<Mat> get_LRHF(Mat image);
	vector<Mat> get_LRHF_Noised(Mat image, int k);

private:
	imgPrc *p_img_prc;
	int p_scale;
	Mat normalize_mat(Mat input_mat, bool norm_lable, int N);
	Mat norm_label(Mat input_mat);
	Mat norm_meanRange(Mat input_mat);
	Mat mean_std_norm(Mat input_mat);
	Mat get_deriv(Mat patch);

};

