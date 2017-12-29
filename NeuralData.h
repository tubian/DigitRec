#pragma once
// NeuralData.h: interface for the CNeuralData class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NEURALDATA_H__B3D7240F_DC2E_4D6E_87E0_B9B205057BF2__INCLUDED_)
#define AFX_NEURALDATA_H__B3D7240F_DC2E_4D6E_87E0_B9B205057BF2__INCLUDED_

#include <afxwin.h>
#include<opencv2/opencv.hpp>
#include<vector>
using std::vector;
using namespace cv;

#define RESAMPLE_LEN 4 // ͼ���ز����Ĳ���
typedef vector<double> iovector; //Ϊ������������һ��˫�����������൱�����������е�һ�У���һ��ѵ����������

class CNeuralData  // ѵ��������
{
public:
	CNeuralData(CString DataPath, int ResampleLength=4, Rect SrcRect= Rect(0,0,32,64));
	virtual ~CNeuralData();

	//�����ĳ�ʼ��
	void Init(Rect SrcRect,int nInputs);

	//�����������
	void Clear();

	void GetClassInfoFromDir(CString strDir); //����strDir�еķ���Ŀ¼��Ϣ���ѵ����������Ŀ¼�����������Ϣ

											  //���ѵ����������Ŀ¼�����������Ϣ
	bool AddData(CString strImgDir, CString strClassName);

	//ȡ������ѵ����������
	vector<vector<double> >& GetInputSet();
	//ȡ�������������
	vector<vector<double> >& GetOutputSet();

	//ȡ��ѵ�������Լ�ѵ�������Ĵ�ȡ·��
	bool CreateTrainingSetFromData();
	bool GetTrainingSet();  //ȡ������ѵ��������������ѵ������������趨��������
	bool GetSamplePaths();  //ȡ��ѵ�������Ĵ�ȡ·��������Щ��Ϣ�����m_vecSamples

	vector<CString> GetClassName() { return m_vecClassNames; }

	//�����������
	CString GetClassName(int nClass);

	//�����������
	int GetClassNum() { return m_nClassNumber; }

	//�������뵥Ԫ��Ŀ
	int GetFeatureDimensions() { return m_nFeatureDimensions; }


protected:

	vector<CString> m_vecClassNames;       //�������(��Ÿ����������ļ�������)
	vector<CString>	m_vecDirs;             //���ѵ�����������Ŀ¼
	vector<vector<CString> > m_vecSamples; //ȫ��ѵ�������ļ��Ĵ�ȡ·��

    // ����ѵ������
	vector<iovector > m_SetOut; //�����������
	vector<iovector > m_SetIn;  //����������������
	int m_nResampleLength;        //�ز�������
	int m_nClassNumber;            //�����Ŀ   
	int m_nFeatureDimensions;      //������������������ά��
	Rect m_rt;                     //ÿ��ͼ��Ĵ�С
};

#endif // !defined(AFX_NEURALDATA_H__B3D7240F_DC2E_4D6E_87E0_B9B205057BF2__INCLUDED_)
