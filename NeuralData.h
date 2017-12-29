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

#define RESAMPLE_LEN 4 // 图像重采样的步长
typedef vector<double> iovector; //为输入输出定义的一个双精度向量，相当于样本矩阵中的一行，即一个训练样本向量

class CNeuralData  // 训练数据类
{
public:
	CNeuralData(CString DataPath, int ResampleLength=4, Rect SrcRect= Rect(0,0,32,64));
	virtual ~CNeuralData();

	//基本的初始化
	void Init(Rect SrcRect,int nInputs);

	//清空内置容器
	void Clear();

	void GetClassInfoFromDir(CString strDir); //根据strDir中的分类目录信息获得训练数据所在目录和类别名称信息

											  //添加训练数据所在目录和类别名称信息
	bool AddData(CString strImgDir, CString strClassName);

	//取得输入训练样本矩阵
	vector<vector<double> >& GetInputSet();
	//取得输出向量矩阵
	vector<vector<double> >& GetOutputSet();

	//取得训练集合以及训练样本的存取路径
	bool CreateTrainingSetFromData();
	bool GetTrainingSet();  //取得整个训练集，包括读入训练样本矩阵和设定输出类编码
	bool GetSamplePaths();  //取得训练样本的存取路径，将这些信息存放至m_vecSamples

	vector<CString> GetClassName() { return m_vecClassNames; }

	//返回类别名称
	CString GetClassName(int nClass);

	//返回类别总数
	int GetClassNum() { return m_nClassNumber; }

	//返回输入单元数目
	int GetFeatureDimensions() { return m_nFeatureDimensions; }


protected:

	vector<CString> m_vecClassNames;       //类别名称(存放该类样本的文件夹名称)
	vector<CString>	m_vecDirs;             //存放训练样本的类别目录
	vector<vector<CString> > m_vecSamples; //全部训练样本文件的存取路径

    // 包含训练数据
	vector<iovector > m_SetOut; //输出向量矩阵
	vector<iovector > m_SetIn;  //输入样本向量矩阵
	int m_nResampleLength;        //重采样步长
	int m_nClassNumber;            //类别数目   
	int m_nFeatureDimensions;      //单个样本特征向量的维数
	Rect m_rt;                     //每个图像的大小
};

#endif // !defined(AFX_NEURALDATA_H__B3D7240F_DC2E_4D6E_87E0_B9B205057BF2__INCLUDED_)
