#ifndef __NEURON_H__
#define __NEURON_H__
#include <afxwin.h>
#define NEURALNET_VERSION 0x03

#define NEED_MOMENTUM //����Ҫ���ӳ�����뽫��һ���ע��ȥ��


typedef double WEIGHT_TYPE; //����Ȩֵ����������

#ifdef NEED_MOMENTUM
#define MOMENTUM  0.3;
#endif
//��Ԫ�ṹ��SNeuron
struct SNeuron
{
	int m_nInput;
	WEIGHT_TYPE *m_pWeights;
#ifdef NEED_MOMENTUM                 //��Ҫ����ʱ������һ��Ȩֵ������¼ǰһ�θ���ʱ��Ȩֵ�仯
	WEIGHT_TYPE *m_pPrevUpdate;

#endif
	double m_dActivation;            //��Ԫ��Ӧ(���)������Sigmoid����������
	double m_dError;                 //��Ԫ�����ֵ
	void Init(int nInput)
	{
		m_nInput = nInput + 1;
		m_pWeights = new WEIGHT_TYPE[m_nInput];

#ifdef NEED_MOMENTUM
		m_pPrevUpdate = new WEIGHT_TYPE[m_nInput];   //Ϊ��һ��Ȩֵ�������ռ�
#endif
		m_dActivation = 0; //��Ԫ��Ӧ(���)������Sigmoid����������
		m_dError = 0; //��Ԫ�����ֵ
	}

	~SNeuron()
	{
		//�ͷſռ�
		delete[]m_pWeights;
#ifdef NEED_MOMENTUM
		delete[]m_pPrevUpdate;
#endif
	}
};


struct SNeuronLayer //�������
{
	/////////////////////////////////����//////////////////////////////
	int m_nNeuron; //�ò����Ԫ��Ŀ
	SNeuron *m_pNeurons; //��Ԫ����
  ////////////////////////////////����////////////////////////////////

	SNeuronLayer(int nNeuron, int nInputsPerNeuron)
	{
		m_nNeuron = nNeuron;
		m_pNeurons = new SNeuron[nNeuron]; //����nNeuron����Ԫ������ռ�

		for (int i = 0; i<nNeuron; i++)
		{
			m_pNeurons[i].Init(nInputsPerNeuron); //��Ԫ��ʼ��
		}
	}
	~SNeuronLayer()
	{
		delete[]m_pNeurons; //�ͷ���Ԫ����
	}
}; //SNeuronLayer

//////////////////����ѵ���ļ�ʱʹ��/////////////
struct NEURALNET_HEADER   //����·��Ϣͷ
{
	DWORD dwVersion; //�汾��Ϣ

	 // ��ʼ�����������ɸ���
	int m_nInput; //����������Ŀ
	int m_nOutput; //���������Ԫ��Ŀ
	int m_nHiddenLayer; //���ز���Ŀ��DigitRec��ֻ֧��1�����ز�

						//����ÿ��ѵ��ǰ���õĲ���
	int m_nNeuronsPerLyr;	//���ز㵥Ԫ��Ŀ	
	int m_nEpochs; //ѵ��ʱ����Ŀ�����򴫲��㷨�ĵ���������
}; //NEURALNET_HEADER


#endif // __NEURON_H__