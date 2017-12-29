// ANN.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include"NeuralNet.h"
#include"NeuralData.h"
using std::cout;
using std::endl;
void Train(CString _DataPath, LPCTSTR savepath);
void Recognize(Mat &src, LPCTSTR NeuralDataPath);

int main()
{

	CString path = "C:/Users/HUA/Documents/Visual Studio 2015/Projects/ANN/ANN/Dataset/Train";
	Mat src = imread("C:/Users/HUA/Documents/Visual Studio 2015/Projects/ANN/ANN/Dataset/Test/4_Verdana.bmp", IMREAD_GRAYSCALE);
	LPCTSTR savepath= _T("C:/Users/HUA/Documents/Visual Studio 2015/Projects/ANN/ANN/Dataset/Train/data.dat");

	Train(path, savepath);
	Recognize(src, savepath);
    return 0;
}


void Train(CString _DataPath, LPCTSTR savepath)
{
	CNeuralData neuraldata(_DataPath);
	CNeuralNet neuralnet(neuraldata.GetFeatureDimensions(), neuraldata.GetClassNum(), 20, 0.1, 300);
	neuralnet.Train(neuraldata.GetInputSet(), neuraldata.GetOutputSet());
    neuralnet.SaveToFile(savepath);
} 


void Recognize(Mat &src, LPCTSTR NeuralDataPath)
{
	CNeuralNet  neuralnet2;
	neuralnet2.LoadFromFile(NeuralDataPath);

	vector<double> vecToMat;
	//如果图像不连续，将待识别图像转换为向量形式opencv
	if (src.isContinuous())
	{
		int rowsmax = src.rows;
		int colsmax = src.cols;
		const uchar *data = src.ptr<uchar>(0);
		for (int i = 0; i<rowsmax*colsmax; i += RESAMPLE_LEN)
		{
			vecToMat.push_back(data[i]);
		}
	}
	else if (!src.isContinuous())
	{
		int rowsmax = src.rows;
		int colsmax = src.cols;
		for (int i = 0; i<rowsmax; i += RESAMPLE_LEN)
		{
			const uchar *data = src.ptr<uchar>(i);
			for (int j = 0; j<colsmax; j += RESAMPLE_LEN)
			{
				vecToMat.push_back(data[j]);
			}
		}
	}
	double confidence = 0;

	int classlabel = neuralnet2.Recognize(vecToMat, confidence);

	cout << "\n识别值：" << classlabel << endl;
	cout <<"\n置信度："<<confidence << endl;
	
}
