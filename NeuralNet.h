#pragma once
#include"Neuron.h"
#include<cstring>
#include <math.h>
#include"NeuralData.h"
#include<opencv2/opencv.hpp>
using namespace cv;
using std::vector;
using std::cout;
using std::endl;

#define BIAS 1 //偏置项w0的系数
#define WEIGHT_FACTOR 0.1 //一个大于 0 小于 1 的浮点数，用来限定初始权值的范围

//返回一个0，1之间的随机浮点数
inline double RandFloat() { return (rand()) / (RAND_MAX + 1.0); }

//返回一个大于 -1 小于 1的随机浮点数
inline double RandomClamped() { return WEIGHT_FACTOR*(RandFloat() - RandFloat()); }

class CNeuralNet
{
private:
	/////// 初始化参数,训练开始至结束过程中不能修改 ////////
	int m_nInput; //输入单元数目
	int m_nOutput; //输出单元数目
	int m_nNeuronsPerLyr; //隐藏层单元数目		 
	int m_nHiddenLayer;   // 隐藏层数目，不包含输出层

	//训练配置信息
	int m_nMaxEpoch;     //最大训练时代数目
	double m_dMinError;  //误差阈值

	///////////////////////////////////
	// 动态参数
	int m_nEpochs;            //训练时代计数
	double m_dLearningRate;   //学习率
	double m_dErrorSum;       //一个时代的累计误差
	double m_dErr;            //一个时代的平均到每一次训练、每个输出的误差

	bool m_bStop;//控制训练过程是否中途停止

	SNeuronLayer *m_pHiddenLyr; //隐藏层
	SNeuronLayer *m_pOutLyr;    //输出层

	vector<double> m_vecError; //训练过程中对应于各个时代的训练误差
public:
	// 构造函数
	CNeuralNet();
	CNeuralNet(int nInput, int nOutput, int nHiddenLyrNeurons,double LearnRate=0.1,int MaxEpoch=200,double ErrorTolerance =0.01);
	~CNeuralNet();

	// 初始化网络
	void InitializeNetwork();

	// 计算网络输出，前向传播
	bool CalculateOutput(vector<double> input, vector<double> &output);

	// 训练一个时代，反向调整
	bool TrainingEpoch(vector<iovector>& inputs, vector<iovector>& outputs);

	bool Train(vector<iovector>& SetIn, vector<iovector>& SetOut);  //整个反向传播训练过程

																   // 识别某一个未知类别样本，返回类别标号
	int Recognize(const vector<double> & InputSample, double &dConfidence);


	// 获取参数
	double GetErrorSum() { return m_dErrorSum; } //返回当前时代误差
	double GetError() { return m_dErr; } //返回平均误差
	int GetEpoch() { return m_nEpochs; } //返回时代数
	int GetNumOutput() { return m_nOutput; } //返回输出层单元数目
	int GetNumInput() { return m_nInput; } //返回输入层单元数目
	int GetNumNeuronsPerLyr() { return m_nNeuronsPerLyr; } //返回隐藏层单元数目

														   // 设定训练配置信息
	void SetMaxEpoch(int nMaxEpoch) { m_nMaxEpoch = nMaxEpoch; }
	void SetMinError(double dMinError) { m_dMinError = dMinError; }
	void SetLearningRate(double dLearningRate) { m_dLearningRate = dLearningRate; }

	void SetStopFlag(bool bStop) { m_bStop = bStop; }

	// 保存和装载训练文件
	bool SaveToFile(LPCTSTR lpszFileName, bool bCreate = true); //保存训练结果
	bool LoadFromFile(LPCTSTR lpszFileName, DWORD dwStartPos = 0); //装载训练结果

protected:

	void CreateNetwork(); //建立网络，为各层单元分配空间


						  // Sigmoid 激励函数
	double	  Sigmoid(double netinput)
	{
		double response = 1.0; //控制函数陡峭程度的参数

		return (1 / (1 + exp(-netinput / response)));
	}
};

