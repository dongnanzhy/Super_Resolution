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
#include "ml_nn.h"
#include "imgPrc.h"
#include "imgIO.h"
//#include "imgDeNoise.h"

using namespace cv;

ml_nn::ml_nn( void )
{
	p_img_prc = nullptr;
	p_scale = 479;
}

ml_nn::~ml_nn( void )
{
	close_ml_nn();
}

int ml_nn::close_ml_nn( void )
{
	if( p_img_prc != nullptr ) {
		p_img_prc = nullptr;
	}
	p_scale = 0;

	return 0;
}

// ----------------------------------------- generate for training data ------------------------------------------------//

// ------------------------------
//   input: a sequence of training images and patch size
//   return: Training data matrix, each row is one instance
//   original method, normalization for whole training data
// ------------------------------
Mat ml_nn::get_train(vector<Mat> images, int N)
{
	int height;
	int width;
	Mat dst;
	dst = Mat(0, N*N+7+1, CV_32FC1);
	std::cout <<"train data generating..." << std::endl;

	for (size_t  k = 0; k < images.size();k++) {
	    vector<Mat> channel_vec = get_LRHF(images[k]);
		// (input image is YCC format)
		Mat yChannel_LR = channel_vec[0]; Mat yChannel_HF = channel_vec[1];
		Mat cb_channel = channel_vec[2]; Mat cr_channel =channel_vec[3];
	    height =  yChannel_LR.rows;
		width =  yChannel_LR.cols;
		for (int i = 0; i <= height-N; i++) {
		    for (int j = 0; j <= width-N; j++) {	
				Mat featureVec = Mat(1, N*N+7+1, CV_32FC1);
				// (get neighbor)
				Mat patch = yChannel_LR(Rect(j, i, N, N)).clone();
				Mat patchVec = patch.reshape(0,1);
				// (get derivatives)
				Mat diffVec = get_deriv(patch);
				// (combine feature)
				patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
				featureVec.at<float>(0,N*N) = cb_channel.at<short>(i+2,j+2);
				featureVec.at<float>(0,N*N+1) = cr_channel.at<short>(i+2,j+2);
				diffVec.copyTo(featureVec(Rect(N*N+2, 0, diffVec.cols, diffVec.rows)));
				featureVec.at<float>(0,N*N+7) = yChannel_HF.at<short>(i+2,j+2);
				// (push feature vector to traning data)
				dst.push_back(featureVec);		
		    }
		    std::cout<< k << " th image, " << i << " th row generated.." << std::endl;
	      }
	}
    dst = normalize_mat(dst, true, N); // for ANN
	//dst = normalize(dst, false, N); // for svm,KNN
    std::cout <<"train data generated..." << std::endl;
	return dst;
}
// ------------------------------
//   input: a sequence of training images and patch size
//   return: Training data matrix, each row is one instance
//   improved method, normalization for each patch
// ------------------------------
Mat ml_nn::get_train_normPatch(vector<Mat> images, int N, bool isNoised)
{
	int height;
	int width;
	Mat dst;
	dst = Mat(0, N*N+9+1, CV_32FC1);
	std::cout <<"train data generating..." << std::endl;
	
	for (size_t  k = 0; k < images.size();k++) {
		vector<Mat> channel_vec(4);
		if (isNoised) {
			channel_vec = get_LRHF_Noised(images[k], k);
		} else {
			channel_vec = get_LRHF(images[k]);
		}
	    
		// (input image is YCC format)
		Mat yChannel_LR, yChannel_HF, cb_channel, cr_channel;
		channel_vec[0].convertTo(yChannel_LR,CV_32FC1); 
		channel_vec[1].convertTo(yChannel_HF,CV_32FC1);
		channel_vec[2].convertTo(cb_channel,CV_32FC1);
		channel_vec[3].convertTo(cr_channel,CV_32FC1);
	    height =  yChannel_LR.rows;
		width =  yChannel_LR.cols;
		for (int i = 0; i <= height-N; i++) {
		    for (int j = 0; j <= width-N; j++) {		
				Mat featureVec = Mat(1, N*N+9+1, CV_32FC1);
				Mat patch = yChannel_LR(Rect(j, i, N, N)).clone();
				float patchMean =float (mean( patch ).val[0]);
				double max, min;
                minMaxLoc(patch, &min, &max);
				float dyRange = float(max - min);
				if (max == min) continue;
				patch = (patch - min) / (max - min);
				Mat patchVec = patch.reshape(0,1);
				Mat diffVec = get_deriv(patch);
				
				minMaxLoc(cb_channel(Rect(j, i, N, N)), &min, &max);
				float cb_val;
				if (cb_channel.at<float>(i+2,j+2) == min) {
					cb_val = 0;
				} else {
				    cb_val = float ((cb_channel.at<float>(i+2,j+2) - min) / (max-min));
				}
				minMaxLoc(cr_channel(Rect(j, i, N, N)), &min, &max);
				float cr_val;
				if (cr_channel.at<float>(i+2,j+2) == min) {
					cr_val = 0;
				} else {
				    cr_val = float ((cr_channel.at<float>(i+2,j+2) - min) / (max-min));
				}
				
				patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
				featureVec.at<float>(0,N*N) = cb_val;
				featureVec.at<float>(0,N*N+1) = cr_val; 
				diffVec.copyTo(featureVec(Rect(N*N+2, 0, diffVec.cols, diffVec.rows)));
				featureVec.at<float>(0,N*N+7) = patchMean/1024;                                        //Y channel original mean value
				featureVec.at<float>(0,N*N+8) = dyRange/1024;                                          //Y channel original dynamic range
				featureVec.at<float>(0,N*N+9) = yChannel_HF.at<float>(i+2,j+2);
				// (push feature vector to traning data)
				dst.push_back(featureVec);
		    }
		    std::cout<< k << " th image, " << i << " th row generated.." << std::endl;
	      }
	} 
	dst = norm_label(dst);
    std::cout <<"train data generated..." << std::endl;
	return dst;
}
// ------------------------------
//   get same test data from training data
// ------------------------------
Mat ml_nn::get_CV_test(Mat train_data)
{
	Mat test_data = train_data(Rect(0, 0, train_data.cols-1, train_data.rows)).clone();
	return test_data;
}

// ----------------------------------------- trian model ------------------------------------------------//

// ------------------------------
//   input: training data with each row an instance, label is last column of training data
//   return: trained NN model
// ------------------------------
CvANN_MLP* ml_nn::nn_model(Mat train_data)
{
   Mat instance = train_data(Rect(0, 0, train_data.cols -1, train_data.rows)).clone();
   Mat label = train_data.col(train_data.cols -1).clone();
   
   CvANN_MLP* nnetwork = new CvANN_MLP;

   // set the network to be 3 layer 30->20->20->1
   // - one input node per attribute in a sample
   // - 25 + 25 hidden nodes
   // - one output node
   int layers_d[] = { instance.cols, 25,25, 1};
   Mat layers = Mat(1,4,CV_32SC1);
   for (int i = 0; i < 4; i++)
   {
	   layers.at<int>(0,i) = layers_d[i];
   }
   nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.2,1);
   // (set the training parameters)
   CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(
          // (terminate the training after either 1000 iterations or a very small change in the network wieghts below the specified value)
          cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.0000000001),
          // (use backpropogation for training)
          CvANN_MLP_TrainParams::BACKPROP,
          // (co-efficents for backpropogation training)
          0.1, 0.1);
   // (train)
   int iterations = nnetwork->train(instance, label, Mat(), Mat(), params, CvANN_MLP::NO_INPUT_SCALE+CvANN_MLP::NO_OUTPUT_SCALE);
   std::cout << "train model success!!" << std::endl << "iter =  " << std::endl << iterations << std::endl;
   return nnetwork;
}
// ------------------------------
//   input: training data with each row an instance, label is last column of training data
//   return: trained SVM model
// ------------------------------
CvSVM* ml_nn::svm_model(Mat train_data)
{
   Mat instance = train_data(Rect(0, 0, train_data.cols -1, train_data.rows)).clone();
   Mat label = train_data.col(train_data.cols -1).clone();
   CvSVM* model_svm = new CvSVM;

   // (define Params)
   CvSVMParams params;
   params.svm_type = CvSVM::NU_SVR;
   params.kernel_type = CvSVM::SIGMOID;

   // (params.degree = 2;  for Poly)
   params.gamma = 200;
   params.coef0 = 0;
   params.C = 0.01;
   params.nu = 0.3;
   params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-15);

   model_svm->train(instance, label, Mat(), Mat(), params);
   return model_svm;
}
// ------------------------------
//   input: training data with each row an instance, label is last column of training data
//   return: trained KNN model
// ------------------------------
CvKNearest* ml_nn::kNN_model(Mat train_data)
{
    Mat instance = train_data(Rect(0, 0, train_data.cols -1, train_data.rows)).clone();
    Mat label = train_data(Rect(train_data.cols -1, 0, 1, train_data.rows)).clone();
	CvKNearest* model_KNN = new CvKNearest;

	model_KNN->train(instance, label, Mat(), true, 500, false);
	return model_KNN;
}
/*
void ml_nn::construct_cnn()
{

    // specify loss-function and optimization-algorithm
    network<mse, adagrad> net;
    //network<cross_entropy, RMSprop> net;

    // add layers
    net << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32in, conv5x5, 1-6 f-maps
        << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28in, 6 f-maps, pool2x2
        << fully_connected_layer<tan_h>(14 * 14 * 6, 120)
        << fully_connected_layer<identity>(120, 10);

    assert(net.in_dim() == 32 * 32);
    assert(net.out_dim() == 10);

    // load MNIST dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;

    parse_mnist_labels("C:\YanProject\tiny-cnn-master\tiny-cnn-master\data\train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("C:\YanProject\tiny-cnn-master\tiny-cnn-master\data\train-images.idx3-ubyte", &train_images);

    // train (50-epoch, 30-minibatch)
    net.train(train_images, train_labels, 30, 50);

    // save
    std::ofstream ofs("weights");
    ofs << net;
}
*/

// ----------------------------------------- perform with trained model ------------------------------------------------//

// ------------------------------
//   input: test data with each row an instance, NN model
//   return: column vector with each element one prediction result
// ------------------------------
Mat ml_nn::perform_cv(Mat test_data, CvANN_MLP* nnetwork)
{
    Mat dst = Mat(0, 1, CV_32SC1);
	Mat result = Mat(1, 1, CV_32FC1);
	Mat test_sample;
	for (int tsample = 0; tsample < test_data.rows; tsample++) {
		 test_sample = test_data.row(tsample);
		 nnetwork->predict(test_sample, result);
		 result = result * p_scale;
		 result.convertTo(result,CV_32SC1);
		 dst.push_back(result);
	}
	return dst;
}
// ------------------------------
//   input: test data with each row an instance, SVM model
//   return: column vector with each element one prediction result
// ------------------------------
Mat ml_nn::perform_cv(Mat test_data, CvSVM* model_svm)
{
	Mat result;
	model_svm->predict(test_data, result);
	result.convertTo(result,CV_16SC1);
	return result;
}
// ------------------------------
//   input: LR image, NN model, patch size
//   return: resolution enhanced image
//   original method normalized by whole training data
// ------------------------------
Mat ml_nn::perform(Mat test_img, CvANN_MLP* nnetwork, int patch_size)
{
	int N = patch_size;
    vector<Mat> channels(3);
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
    Mat yChannel = channels[0];
	Mat cb_channel = channels[1];
    Mat cr_channel = channels[2];
	int height =  yChannel.rows;
	int width =  yChannel.cols;

	// (get test data)
	Mat test_data, test_result, tmp;
	test_result = Mat(0, 1, CV_32SC1);
	tmp = Mat(1, 1, CV_32FC1);
	test_data = Mat(0, N*N+7, CV_32FC1);
	
	for (int i = 0; i <= height-N; i++) {
		for (int j = 0; j <= width-N; j++) {	
			Mat featureVec = Mat(1, N*N+7, CV_32FC1);
			// (get neighbor)
			Mat patch = yChannel(Rect(j, i, N, N)).clone();
			Mat patchVec = patch.reshape(0,1);
			// (get derivatives)
			Mat diffVec = get_deriv(patch);
			// (combine feature)
			patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
			featureVec.at<float>(0,N*N) = cb_channel.at<short>(i+2,j+2);
			featureVec.at<float>(0,N*N+1) = cr_channel.at<short>(i+2,j+2);
			diffVec.copyTo(featureVec(Rect(N*N+2, 0, diffVec.cols, diffVec.rows)));
			test_data.push_back(featureVec);
		}
		std::cout << "test data " << i << " th row completed.." << std::endl;
	}
	test_data = normalize_mat(test_data, false, patch_size);
    std::cout << "test data completed" << std::endl;
	for (int tsample = 0; tsample < test_data.rows; tsample++) {
		 nnetwork->predict(test_data.row(tsample), tmp);
		 // (scale value, changable)
		 tmp = tmp * p_scale;
		 tmp.convertTo(tmp,CV_32SC1);
		 test_result.push_back(tmp);;
		 if (tsample%(width-N) == 0) {
			 std::cout << tsample/(width-N) << " th row completed.." << std::endl;
		 }
	}
	// (restore HF component)
	Mat HF_tmp = test_result.reshape(0,height-N+1);
	HF_tmp.convertTo(HF_tmp,CV_16SC1);
	Mat HF = Mat::zeros(height,width,CV_16SC1);
	HF_tmp.copyTo(HF(Rect(2,2,width-N+1,height-N+1)));
	std::cout << "compute HF success!! " << std::endl;
	// (restore Ychannel)
	Mat yChannel_restore;
	add(yChannel, HF, yChannel_restore,noArray(),CV_16SC1);
	std::cout << "restore yChannel success!! " << std::endl;
	imgIO *p_img_io = new imgIO();
	p_img_io->xmlWrite(HF,"restored HF.xml");
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image)
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
    return out_rgb;
}
// ------------------------------
//   input: LR image, NN model, patch size
//   return: resolution enhanced image
//   new method normalized by patch, for evaluation
// ------------------------------
Mat ml_nn::perform_normPatch(Mat test_img, CvANN_MLP* nnetwork, int patch_size)
{
    int N = patch_size;
	test_img.convertTo(test_img,CV_32FC3);
    vector<Mat> channels(3);
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
	Mat yChannel, cb_channel, cr_channel;
    channels[0].convertTo(yChannel,CV_32FC1);
	channels[1].convertTo(cb_channel,CV_32FC1);
    channels[2].convertTo(cr_channel,CV_32FC1);
	int height =  yChannel.rows;
	int width =  yChannel.cols;

	// (get test data)
	Mat test_data, test_result, tmp;
	test_result = Mat(0, 1, CV_32FC1);
	tmp = Mat(1, 1, CV_32FC1);
	test_data = Mat(0, N*N+9, CV_32FC1);
	
	for (int i = 0; i <= height-N; i++) {
		for (int j = 0; j <= width-N; j++) {	
			Mat featureVec = Mat(1, N*N+9, CV_32FC1);		
	        Mat patch = yChannel(Rect(j, i, N, N)).clone();
			float patch_mean =float (mean( patch ).val[0]);
			double max, min;
            minMaxLoc(patch, &min, &max);
			float dyRange = float(max - min);
			if (max == min) {
				test_data.push_back(Mat::zeros(1,N*N+9,CV_32FC1));
				continue;
			}
			patch = (patch - min) / (max - min);
			Mat patchVec = patch.reshape(0,1);
			Mat diffVec = get_deriv(patch);
			
			minMaxLoc(cb_channel(Rect(j, i, N, N)), &min, &max);
			float cb_val;
			if (cb_channel.at<float>(i+2,j+2) == min) {
				cb_val = 0;
			} else {
				cb_val = float ((cb_channel.at<float>(i+2,j+2) - min) / (max-min));
			}
			minMaxLoc(cr_channel(Rect(j, i, N, N)), &min, &max);
			float cr_val;
			if (cr_channel.at<float>(i+2,j+2) == min) {
				cr_val = 0;
			} else {
				cr_val = float ((cr_channel.at<float>(i+2,j+2) - min) / (max-min));
			}
			
			patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
			featureVec.at<float>(0,N*N) = cb_val;
			featureVec.at<float>(0,N*N+1) = cr_val;
			diffVec.copyTo(featureVec(Rect(N*N+2, 0, diffVec.cols, diffVec.rows)));
			featureVec.at<float>(0,N*N+7) = patch_mean/1024;                                    //Y channel original mean value
		    featureVec.at<float>(0,N*N+8) = dyRange/1024;                                       //Y channel original dynamic range
			test_data.push_back(featureVec);
		}
		std::cout << "test data " << i << " th row completed.." << std::endl;
	}
	
    std::cout << "test data completed" << std::endl;
	std::cout <<test_data.row(50) << std::endl;
	for (int tsample = 0; tsample < test_data.rows; tsample++) {
		 nnetwork->predict(test_data.row(tsample), tmp);
		 if (tmp.at<float>(0,0) > 1) {
			 tmp.at<float>(0,0) = float(0.8);;
		 } 
		  test_result.push_back(tmp); 
		 if (tsample%(width-N) == 0) {
			 std::cout << tsample/(width-N) << " th row completed.." << std::endl;
		 }
	}
	test_result = test_result*p_scale;
	// (restore HF component)
	Mat HF_tmp = test_result.reshape(0,height-N+1);
	Mat HF = Mat::zeros(height,width,CV_32FC1);
	HF_tmp.copyTo(HF(Rect(2,2,width-N+1,height-N+1)));
	std::cout << "compute HF success!! " << std::endl;
	// (restore Ychannel)
	Mat yChannel_restore;
	add(yChannel, HF, yChannel_restore,noArray(),CV_32FC1);
	std::cout << "restore yChannel success!! " << std::endl;
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image)
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
	HF.convertTo(HF, CV_16SC1);
	imgIO *p_img_io = new imgIO();
	p_img_io->xmlWrite(HF,"restored HF.xml");
	out_rgb.convertTo(out_rgb, CV_16UC3);
    return out_rgb;
}
// ------------------------------
//   input: LR 4K image, NN model, patch size
//   return: resolution enhanced image
//   final model to get HR 4K image
// ------------------------------
Mat ml_nn::perform_4K(Mat test_img, CvANN_MLP* nnetwork, int patch_size)
{
    int N = patch_size;
	test_img.convertTo(test_img,CV_32FC3);
    vector<Mat> channels(3);
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
	Mat yChannel, cb_channel, cr_channel;
    channels[0].convertTo(yChannel,CV_32FC1);
	channels[1].convertTo(cb_channel,CV_32FC1);
    channels[2].convertTo(cr_channel,CV_32FC1);
	int height =  yChannel.rows;
	int width =  yChannel.cols;

	// (test result)
	Mat test_result;
	test_result = Mat(0, 1, CV_32FC1);
	// (prepare for test data)
	for (int i = 0; i <= height-N; i++) {
		for (int j = 0; j <= width-N; j++) {	
			Mat featureVec = Mat(1, N*N+9, CV_32FC1);
		    Mat tmp = Mat(1, 1, CV_32FC1);
	        Mat patch = yChannel(Rect(j, i, N, N)).clone();
			float patch_mean =float (mean( patch ).val[0]);
			double max, min;
            minMaxLoc(patch, &min, &max);
			float dyRange = float(max - min);
			if (max == min) {
				tmp.at<float>(0,0);
				test_result.push_back(tmp); 
				continue;
			}
			patch = (patch - min) / (max - min);
			Mat patchVec = patch.reshape(0,1);
			Mat diffVec = get_deriv(patch);
			
			minMaxLoc(cb_channel(Rect(j, i, N, N)), &min, &max);
			float cb_val;
			if (cb_channel.at<float>(i+2,j+2) == min) {
				cb_val = 0;
			} else {
				cb_val = float ((cb_channel.at<float>(i+2,j+2) - min) / (max-min));
			}
			minMaxLoc(cr_channel(Rect(j, i, N, N)), &min, &max);
			float cr_val;
			if (cr_channel.at<float>(i+2,j+2) == min) {
				cr_val = 0;
			} else {
				cr_val = float ((cr_channel.at<float>(i+2,j+2) - min) / (max-min));
			}
			
			patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
			featureVec.at<float>(0,N*N) = cb_val;
			featureVec.at<float>(0,N*N+1) = cr_val;
			diffVec.copyTo(featureVec(Rect(N*N+2, 0, diffVec.cols, diffVec.rows)));
			featureVec.at<float>(0,N*N+7) = patch_mean/1024;                                    //Y channel original mean value
		    featureVec.at<float>(0,N*N+8) = dyRange/1024;                                       //Y channel original dynamic range

			nnetwork->predict(featureVec, tmp);
			if (tmp.at<float>(0,0) > 1) {
			   tmp.at<float>(0,0) = float(0.8);
		    } 
		    test_result.push_back(tmp); 
		}
		std::cout << "test data " << i << " th row completed.." << std::endl;
	}
	test_result = test_result*p_scale;
	// (restore HF component)
	Mat HF_reshaped = test_result.reshape(0,height-N+1);
	Mat HF = Mat::zeros(height,width,CV_32FC1);
	HF_reshaped.copyTo(HF(Rect(2,2,width-N+1,height-N+1)));
    // (Manually denoise, can reduce little noise)
	/*
	Mat HF_tmp = Mat::zeros(HF.size(),HF.type());
	for (auto i = 0; i <= HF_tmp.rows-N; i++)
	{
		for (auto j = 0; j <= HF_tmp.cols-N; j++)
		{
			float tmp = HF.at<float>(i+N/2,j+N/2);
			Mat patch = HF(Rect(j, i, N, N)).clone();
			double max, min;
            minMaxLoc(patch, &min, &max);
			float dyRange = float(max - min);
			if (dyRange > 200) {
				HF_tmp.at<float>(i+N/2,j+N/2) = 0;
			} else {
				HF_tmp.at<float>(i+N/2,j+N/2) = tmp;
			}
		}
	}
	*/
	Mat HF_tmp = HF;
	std::cout << "compute HF success!! " << std::endl;
	// (restore Ychannel)
	Mat yChannel_restore;
	add(yChannel, HF_tmp, yChannel_restore,noArray(),CV_32FC1);
	std::cout << "restore yChannel success!! " << std::endl;
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image)
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
	imgIO *p_img_io = new imgIO();
	p_img_io->xmlWrite(HF_tmp,"restored HF.xml");
	out_rgb.convertTo(out_rgb, CV_16UC3);
    return out_rgb;
}

// ------------------------------
//   input: LR image, SVM model, patch size
//   return: resolution enhanced image
//   normalize for whole test data, perform for evaluation
// ------------------------------
Mat ml_nn::perform(Mat test_img, CvSVM* model_svm, int patch_size)
{
	int N = patch_size;
    vector<Mat> channels(3);
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
    Mat yChannel = channels[0];
	Mat cb_channel = channels[1];
    Mat cr_channel = channels[2];
	int height =  yChannel.rows;
	int width =  yChannel.cols;

    // (get test data)
	Mat test_data, test_result;
	test_data = Mat(0, N*N+7, CV_32FC1);
	for (int i = 0; i <= height-N; i++) {
		for (int j = 0; j <= width-N; j++) {
			Mat featureVec = Mat(1, N*N+7, CV_32FC1);
			// (get neighbor)
			Mat patch = yChannel(Rect(j, i, N, N)).clone();
			Mat patchVec = patch.reshape(0,1);
			// (get derivatives)
			Mat diffVec = get_deriv(patch);
			// (combine feature)
			patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
			diffVec.copyTo(featureVec(Rect(N*N, 0, diffVec.cols, diffVec.rows)));
			featureVec.at<float>(0,N*N+5) = cb_channel.at<short>(i+2,j+2);
			featureVec.at<float>(0,N*N+6) = cr_channel.at<short>(i+2,j+2);
			test_data.push_back(featureVec);

		}
		std::cout << "test data " << i << " th row completed.." << std::endl;
	}
	test_data = normalize_mat(test_data, false, patch_size);
    std::cout << "test data completed" << std::endl;

    model_svm->predict(test_data, test_result);

	Mat HF_tmp = test_result.reshape(0,height-N+1);
	HF_tmp.convertTo(HF_tmp,CV_16SC1);
	Mat HF = Mat::zeros(height,width,CV_16SC1);
	HF_tmp.copyTo(HF(Rect(2,2,width-N+1,height-N+1)));
	std::cout << "compute HF success!! " << std::endl;
	// (restore Ychannel)
	Mat yChannel_restore;
	add(yChannel, HF, yChannel_restore,noArray(),CV_16SC1);
	std::cout << "restore yChannel success!! " << std::endl;
	imgIO *p_img_io = new imgIO();
	p_img_io->xmlWrite(HF,"restored HF.xml");
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image)
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
    return out_rgb;
}
// ------------------------------
//   input: LR image, KNN model, patch size
//   return: resolution enhanced image
//   normalize for whole test data, perform for evaluation
// ------------------------------
Mat ml_nn::perform(Mat test_img, CvKNearest* model_KNN, int patch_size)
{
	int N = patch_size;
	const int K = 10;
    vector<Mat> channels(3);
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
    Mat yChannel = channels[0];
	Mat cb_channel = channels[1];
    Mat cr_channel = channels[2];
	int height =  yChannel.rows;
	int width =  yChannel.cols;

	// (get test data)
	Mat test_data;
	test_data = Mat(0, N*N+7, CV_32FC1);
	
	for (int i = 0; i <= height-N; i++) {
		for (int j = 0; j <= width-N; j++) {	
			Mat featureVec = Mat(1, N*N+7, CV_32FC1);
			// (get neighbor)
			Mat patch = yChannel(Rect(j, i, N, N)).clone();
			Mat patchVec = patch.reshape(0,1);
			// (get derivatives)
			Mat diffVec = get_deriv(patch);
			// (combine feature)
			patchVec.copyTo(featureVec(Rect(0, 0, patchVec.cols, patchVec.rows)));
			diffVec.copyTo(featureVec(Rect(N*N, 0, diffVec.cols, diffVec.rows)));
			featureVec.at<float>(0,N*N+5) = cb_channel.at<short>(i+2,j+2);
			featureVec.at<float>(0,N*N+6) = cr_channel.at<short>(i+2,j+2);
			test_data.push_back(featureVec);
		}
		std::cout << "test data " << i << " th row completed.." << std::endl;
	}
	test_data = normalize_mat(test_data,false, patch_size);
    std::cout << "test data completed" << std::endl;

	Mat test_result = Mat(test_data.rows, 1, test_data.type());
	Mat neighborResponses = Mat(test_data.rows, K, test_data.type());
	Mat dist = Mat(test_data.rows, K, test_data.type());

	model_KNN->find_nearest(test_data,K,test_result,neighborResponses,dist);
	// (restore HF component)
	Mat HF_tmp = test_result.reshape(0,height-N+1);
	HF_tmp.convertTo(HF_tmp,CV_16SC1);
	Mat HF = Mat::zeros(height,width,CV_16SC1);
	HF_tmp.copyTo(HF(Rect(2,2,width-N+1,height-N+1)));
	std::cout << "compute HF success!! " << std::endl;
	// (restore Ychannel)
	Mat yChannel_restore;
	add(yChannel, HF, yChannel_restore,noArray(),CV_16SC1);
	std::cout << "restore yChannel success!! " << std::endl;
	imgIO *p_img_io = new imgIO();
	p_img_io->xmlWrite(HF,"restored HF.xml");
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image) 
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
    return out_rgb;
}

// ----------------------------------------- model I/O ------------------------------------------------//

//--------------------------------
// write ANN model to xml file
//--------------------------------
void ml_nn::model_write(CvANN_MLP* nnetwork, char* filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);
	nnetwork->write(*fs, "NN_Model"); 
	fs.release();
}
//--------------------------------
// write SVM model to xml file
//--------------------------------
void ml_nn::model_write(CvSVM* model_svm, char* filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);
	model_svm->write(*fs, "SVM_Model"); 
	fs.release();
}
//--------------------------------
// write KNN model
//--------------------------------
void ml_nn::model_write(CvKNearest* model_KNN, char* filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);
	model_KNN->write(*fs, "KNN_Model"); 
	fs.release();
}
//--------------------------------
// read ANN model from xml file
//--------------------------------
CvANN_MLP* ml_nn::model_read_nn(char* filePath)
{
	FileStorage fs(filePath, FileStorage::READ);
	CvANN_MLP* nnetwork = new CvANN_MLP;
    nnetwork->load(filePath, "NN_Model");
	fs.release();
	return nnetwork;
}
//--------------------------------
// read SVM model from xml file
//--------------------------------
CvSVM* ml_nn::model_read_svm(char* filePath)
{
	FileStorage fs(filePath, FileStorage::READ);
	CvSVM* model_svm = new CvSVM;
    model_svm->load(filePath, "SVM_Model");
	fs.release();
	return model_svm;
}
//--------------------------------
// read KNN model from xml file
//--------------------------------
CvKNearest* ml_nn::model_read_KNN(char* filePath)
{
	FileStorage fs(filePath, FileStorage::READ);
	CvKNearest* model_KNN = new CvKNearest;
    model_KNN->load(filePath, "KNN_Model");
	fs.release();
	return model_KNN;
}

// ----------------------------------------- util functions ------------------------------------------------//

//--------------------------------
// input: LR image and HF components
// return: HR image
// restore HR image from HF components, for error analysis
//--------------------------------
Mat ml_nn::restore_from_hf(Mat test_img, Mat HF)
{
	vector<Mat> channels(3);
	test_img.convertTo(test_img, CV_32FC3);
	HF.convertTo(HF, CV_32FC1);
	// (Stretch) 
	//HF = HF*1.5;
	// (Wavelet Denoise)
	//imgDeNoise *p_img_denoise = new imgDeNoise();
	//Mat HF_tmp = p_img_denoise->wavelet_denoise_HF(HF,100);
	// (Bilatera Filter)
	//int ksize = 7;
	//Mat HF_tmp = Mat(HF.size(), HF.type());
	//bilateralFilter ( HF, HF_tmp, ksize, ksize*2, ksize/2 );
	// (Manually remove noise)	
	Mat HF_tmp = Mat::zeros(HF.size(),HF.type());
	int N = 5;
	for (auto i = 0; i <= HF_tmp.rows-N; i++) {
		for (auto j = 0; j <= HF_tmp.cols-N; j++) {
			float tmp = HF.at<float>(i+N/2,j+N/2);
			Mat patch = HF(Rect(j, i, N, N)).clone();
			double max, min;
            minMaxLoc(patch, &min, &max);
			float dyRange = float(max - min);
			if (dyRange > 200) {
				HF_tmp.at<float>(i+N/2,j+N/2) = 0;
			} else {
				HF_tmp.at<float>(i+N/2,j+N/2) = tmp;
			}
		}
	}
    Mat ycc_img = p_img_prc->rgb2Ycc_cv(test_img);
    split(ycc_img, channels);
    Mat yChannel = channels[0];
	Mat yChannel_restore;
	add(yChannel, HF_tmp, yChannel_restore,noArray(),CV_32FC1);
	yChannel_restore.convertTo(yChannel_restore, channels[1].type());
	channels[0] = yChannel_restore;
	// (merge channels back)
	Mat out_ycc;
	merge(channels,out_ycc);
    std::cout << "merge channels success!! " << std::endl;
	// (restore rgb image)
	Mat out_rgb = p_img_prc->ycc2Rgb_cv(out_ycc);
	out_rgb.convertTo(out_rgb, CV_16UC3);
	imgIO *p_img_io = new imgIO();
	HF_tmp.convertTo(HF_tmp, CV_16SC1);
	p_img_io->xmlWrite(HF_tmp,"scaled HF.xml");
	p_img_io->imgWrite(HF_tmp,"HF.pgm");
    return out_rgb;
}
//--------------------------------
// input: matrix needed to be normalized, whether normalize label comlumn(last column), patch size
// return: normalized matrix
// prepare for normalization to whole data
//--------------------------------
Mat ml_nn::normalize_mat(Mat input_mat, bool norm_lable, int N)
{
	Mat tmp1 = Mat(input_mat.rows, N*N, input_mat.type());
	Mat tmp2 = Mat(input_mat.rows, 2, input_mat.type());
	Mat tmp3 = Mat(input_mat.rows, 5, input_mat.type());
	normalize(input_mat(Rect(0, 0, N*N, input_mat.rows)), tmp1, -1.0, 1.0, CV_MINMAX, -1);
	normalize(input_mat(Rect(N*N, 0, 2, input_mat.rows)), tmp2, -1.0, 1.0, CV_MINMAX, -1);
	normalize(input_mat(Rect(N*N+2, 0, 5, input_mat.rows)), tmp3, -1.0, 1.0, CV_MINMAX, -1);

    tmp1.copyTo(input_mat(Rect(0, 0, N*N, input_mat.rows)));
    tmp2.copyTo(input_mat(Rect(N*N, 0, 2, input_mat.rows)));
	tmp3.copyTo(input_mat(Rect(N*N+2, 0, 5, input_mat.rows)));
	
	if (input_mat.cols == N*N+7 || !norm_lable) {
		return input_mat;
	} else {
		double min4, max4;
        Mat tmp_mat4 = input_mat.col(input_mat.cols-1);
        minMaxLoc(tmp_mat4, &min4, &max4);
	    max4 = std::max(max4,std::abs(min4));
        Mat tmp4 = tmp_mat4/(5*max4);
		tmp4.copyTo(input_mat(Rect(N*N+7, 0, 1, input_mat.rows)));
		return input_mat;
	}
}
//--------------------------------
// input: matrix needed to be normalized, patch size
// return: normalized matrix
// normalize label(last comlumn)
//--------------------------------
Mat ml_nn::norm_label(Mat input_mat)
{
	   double min, max;
       Mat tmp_mat = input_mat.col(input_mat.cols-1);
       minMaxLoc(tmp_mat, &min, &max);
	   std::cout<< "max = "<< max << std::endl<<"min = " << min << std::endl;
	   max = std::max(max,std::abs(min));
       Mat tmp = tmp_mat/max;
	   tmp.copyTo(input_mat(Rect(input_mat.cols-1, 0, 1, input_mat.rows)));
	   return input_mat;
}
//--------------------------------
// input: matrix needed to be normalized
// return: normalized matrix
// normalize to [0,1] by whole matrix
//--------------------------------
Mat ml_nn::norm_meanRange(Mat input_mat)
{
	int N = 5;
	Mat tmp1 = Mat(input_mat.rows, 1, input_mat.type());
	Mat tmp2 = Mat(input_mat.rows, 1, input_mat.type());
	normalize(input_mat(Rect(N*N+7, 0, 1, input_mat.rows)), tmp1, 0, 1.0, CV_MINMAX, -1);
	normalize(input_mat(Rect(N*N+8, 0, 1, input_mat.rows)), tmp2, 0, 1.0, CV_MINMAX, -1);

    tmp1.copyTo(input_mat(Rect(N*N+7, 0, 1, input_mat.rows)));
    tmp2.copyTo(input_mat(Rect(N*N+8, 0, 1, input_mat.rows)));
	return input_mat;
}
//--------------------------------
// input: matrix needed to be normalized
// return: normalized matrix
// mean/norm normalization by whole matrix
//--------------------------------
Mat ml_nn::mean_std_norm(Mat input_mat)
{
	Scalar tmp = mean(input_mat);
    input_mat = input_mat - tmp[0];
	return input_mat / norm(input_mat,NORM_L2,noArray());
}
//--------------------------------
// input: patch needed to be calculated
// return: row vector of length 5
// calculate first and second order within patch
//--------------------------------
Mat ml_nn::get_deriv(Mat patch)
{
	Mat grad;
	float x,y,x2,y2,xy;
	int scale = 1;
    int delta = 0;
	int ddepth = CV_32F;
	int x_order,y_order;
	int center = patch.rows / 2;
	// (dP/dx)
    x_order = 1;
	y_order = 0;
	Sobel(patch, grad, ddepth, x_order, y_order, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	x = grad.at<float>(center,center);
	// (dP/dy)
	x_order = 0;
	y_order = 1;
	Sobel(patch, grad, ddepth, x_order, y_order, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	y = grad.at<float>(center,center);
    // (d2P/dx2)
	x_order = 2;
	y_order = 0;
	Sobel(patch, grad, ddepth, x_order, y_order, 5, scale, delta, BORDER_DEFAULT);
	x2 = grad.at<float>(center,center);
    // (d2P/dy2)
	x_order = 0;
	y_order = 2;
	Sobel(patch, grad, ddepth, x_order, y_order, 5, scale, delta, BORDER_DEFAULT);
	y2 = grad.at<float>(center,center);
	// (d2P/dxy)
	x_order = 1;
	y_order = 1;
	Sobel(patch, grad, ddepth, x_order, y_order, 5, scale, delta, BORDER_DEFAULT);
	xy = grad.at<float>(center,center);
	return (Mat_<float>(1,5) << x, y, x2, y2, xy);
}
//--------------------------------
// input: original image prepared for training data
// return: sequence of matrix representing LR, HF, Cb, Cr
// generate LR, HF matrices preparing for training data
//--------------------------------
vector<Mat> ml_nn::get_LRHF(Mat image)
{
   vector<Mat> channel_vec(4);
   image.convertTo(image,CV_32FC3);
   Mat LR;
   GaussianBlur( image, LR, Size(7,7), 1.4, 1.4);   // reduce the PSNR to 40
   // (to YCbCr)
   Mat ycc_img = p_img_prc->rgb2Ycc_cv(image);
   Mat ycc_LR = p_img_prc->rgb2Ycc_cv(LR);
   // (get YChannel)
   vector<Mat> channels_img(3);
   split(ycc_img, channels_img);
   Mat yChannel_img = channels_img[0];
   vector<Mat> channels_LR(3);
   split(ycc_LR, channels_LR);
   Mat yChannel_LR = channels_LR[0];
   // (compute HF)
   Mat yChannel_HF;
   subtract(yChannel_img, yChannel_LR, yChannel_HF,noArray(),CV_32FC1);
   channel_vec[0] = yChannel_LR;
   channel_vec[1] = yChannel_HF;
   channel_vec[2] = channels_LR[1];
   channel_vec[3] = channels_LR[2];
   return channel_vec;
}
//--------------------------------
// input: original image prepared for training data, image index to find noised LR image
// return: sequence of matrix representing LR, HF, Cb, Cr
// generate LR, HF matrices preparing for training data with noise
//--------------------------------
vector<Mat> ml_nn::get_LRHF_Noised(Mat image, int k)
{
   string source = "C:\\YanProject\\Super_Resolution\\sources\\Source\\Noised_Data\\";
   string path = source + format("%04d", k) + "_LR.ppm";
   imgIO *p_img_io = new imgIO();
   Mat LR = p_img_io->imgRead(path);
   LR = LR / 64;
   LR.convertTo(LR,CV_32FC3);

   vector<Mat> channel_vec(4);
   image.convertTo(image,CV_32FC3);
   Mat ycc_img = p_img_prc->rgb2Ycc_cv(image);
   Mat ycc_LR = p_img_prc->rgb2Ycc_cv(LR);
   // (get YChannel)
   vector<Mat> channels_img(3);
   split(ycc_img, channels_img);
   Mat yChannel_img = channels_img[0];
   vector<Mat> channels_LR(3);
   split(ycc_LR, channels_LR);
   Mat yChannel_LR = channels_LR[0];
   // (compute HF)
   Mat yChannel_HF;
   subtract(yChannel_img, yChannel_LR, yChannel_HF,noArray(),CV_32FC1);
   channel_vec[0] = yChannel_LR;
   channel_vec[1] = yChannel_HF;
   channel_vec[2] = channels_LR[1];
   channel_vec[3] = channels_LR[2];
   return channel_vec;
}