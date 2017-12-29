#ifndef __NEURON_H__
#define __NEURON_H__
#include <afxwin.h>
#define NEURALNET_VERSION 0x03

#define NEED_MOMENTUM //如需要增加冲量项，请将这一项的注释去掉


typedef double WEIGHT_TYPE; //定义权值的数据类型

#ifdef NEED_MOMENTUM
#define MOMENTUM  0.3;
#endif
//神经元结构体SNeuron
struct SNeuron
{
	int m_nInput;
	WEIGHT_TYPE *m_pWeights;
#ifdef NEED_MOMENTUM                 //需要冲量时，定义一个权值变量记录前一次更新时的权值变化
	WEIGHT_TYPE *m_pPrevUpdate;

#endif
	double m_dActivation;            //神经元响应(输出)，经过Sigmoid激励函数后
	double m_dError;                 //神经元的误差值
	void Init(int nInput)
	{
		m_nInput = nInput + 1;
		m_pWeights = new WEIGHT_TYPE[m_nInput];

#ifdef NEED_MOMENTUM
		m_pPrevUpdate = new WEIGHT_TYPE[m_nInput];   //为上一次权值数组分配空间
#endif
		m_dActivation = 0; //神经元响应(输出)，经过Sigmoid激励函数后
		m_dError = 0; //神经元的误差值
	}

	~SNeuron()
	{
		//释放空间
		delete[]m_pWeights;
#ifdef NEED_MOMENTUM
		delete[]m_pPrevUpdate;
#endif
	}
};


struct SNeuronLayer //神经网络层
{
	/////////////////////////////////数据//////////////////////////////
	int m_nNeuron; //该层的神经元数目
	SNeuron *m_pNeurons; //神经元数组
  ////////////////////////////////方法////////////////////////////////

	SNeuronLayer(int nNeuron, int nInputsPerNeuron)
	{
		m_nNeuron = nNeuron;
		m_pNeurons = new SNeuron[nNeuron]; //分配nNeuron个神经元的数组空间

		for (int i = 0; i<nNeuron; i++)
		{
			m_pNeurons[i].Init(nInputsPerNeuron); //神经元初始化
		}
	}
	~SNeuronLayer()
	{
		delete[]m_pNeurons; //释放神经元数组
	}
}; //SNeuronLayer

//////////////////保存训练文件时使用/////////////
struct NEURALNET_HEADER   //神经网路信息头
{
	DWORD dwVersion; //版本信息

	 // 初始化参数，不可更改
	int m_nInput; //网络输入数目
	int m_nOutput; //网络输出单元数目
	int m_nHiddenLayer; //隐藏层数目，DigitRec中只支持1个隐藏层

						//可在每次训练前设置的参数
	int m_nNeuronsPerLyr;	//隐藏层单元数目	
	int m_nEpochs; //训练时代数目（反向传播算法的迭代次数）
}; //NEURALNET_HEADER


#endif // __NEURON_H__