// NeuralData.cpp: implementation of the CNeuralData class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "NeuralData.h"


#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


CNeuralData::CNeuralData(CString DataPath,int ResampleLength,Rect SrcRect)
{
	m_nResampleLength = ResampleLength;
	Init(SrcRect, int(SrcRect.width*SrcRect.height/ResampleLength));
	GetClassInfoFromDir(DataPath);  
	CreateTrainingSetFromData();
}

CNeuralData::~CNeuralData()
{

}

void CNeuralData::Clear()
{

	m_vecDirs.clear();
	m_vecClassNames.clear();
	m_vecSamples.clear();

	m_nClassNumber = 0;
}

void CNeuralData::Init(Rect SrcRect,int FeatureNumber)
{
	m_SetOut.clear();
	m_SetIn.clear();
	Clear();                                //清空各个容器
	m_rt = SrcRect;
	m_nFeatureDimensions = FeatureNumber;   //设定输入数目
}

/*
GetClassInfoFromDir 每个同种类的训练样本放在同一个以类别名称为名的文件夹中，所有不同种类样本的文件件又放在一个地址为strDir的总文件夹中，
                    根据总文件夹中子文件的名称信息获得训练数据所在目录和类别名称信息,类别数量。
strDir：存放所有训练样本的总文件夹地址
*/
void CNeuralData::GetClassInfoFromDir(CString strDir)  
{
	if (strDir.Right(1) != "\\")
		strDir = strDir + "\\";

	CString strDirName; //分类目录名称（类别名）

						//查找训练集合文件夹下的分类目录
	WIN32_FIND_DATA findData;
	HANDLE hFindFile;

	CString strDirFind = strDir;
	strDirFind += "*.*";
	hFindFile = ::FindFirstFile(strDirFind, &findData);

	Clear(); //清空目录容器

	if (hFindFile != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (findData.cFileName[0] == '.')  //..表示当前目录或上级目录。（判定文件名首字母是不是"."）
				continue;
			if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))//判断是否为文件夹（目标文件标记，文件夹标记16），当不是文件夹时跳过本次循环
				continue;

			strDirFind = strDir + findData.cFileName;

			strDirFind += "\\";

			strDirName = findData.cFileName;
			AddData(strDirFind, strDirName);

		} while (::FindNextFile(hFindFile, &findData));//当前目录是否还存在“下一个目录”或者“文件”

		::FindClose(hFindFile);
	}
	else
	{
		AfxMessageBox(_T("没有找到训练样本目录，从而无法提取类别名称信息!"));
		return;
	}
}

/*
AddData根据样本路径信息，文件夹归类信息添加需要识别的类别，并累加类别数量
strImgDir：表示某一类样本图片的文件夹地址
strClassName：表示某一类样本的类别名称
*/
bool CNeuralData::AddData(CString strImgDir, CString strClassName)
{
	//添加样本文件所在目录的路径信息
	m_vecDirs.push_back(strImgDir);

	//添加代表各个类的名字，我们以存放该类的文件夹的名字作为类的名字
	m_vecClassNames.push_back(strClassName);

	m_nClassNumber++; //类别数目
	return true;
}

vector<iovector >& CNeuralData::GetInputSet()
{
	return m_SetIn;
}

vector<iovector >& CNeuralData::GetOutputSet()
{
	return m_SetOut;
}

CString CNeuralData::GetClassName(int nClass)
{
	if (nClass < m_vecClassNames.size())
		return m_vecClassNames[nClass];
	else
		return "";
}

bool CNeuralData::CreateTrainingSetFromData()
{
	//取得训练样本的存取路径
	if (GetSamplePaths() == false)
		return false;

	//取得整个训练集
	if (!GetTrainingSet())
		return false;

	return true;
}

bool CNeuralData::GetSamplePaths()
{
	int nClass = m_vecDirs.size(); //类别数目

	for (int i = 0; i<nClass; i++)
	{
		// 遍历第i类类别目录，将这些样本文件的存取路径保存至 vecFiles
		vector<CString> vecFiles; //某一类训练样本的存取路径

								  // 查找目标为第i类类别目录下的全部 .bmp 图像文件
		CString strToFind = m_vecDirs[i];
		strToFind += "*.bmp";

		WIN32_FIND_DATA findData;
		HANDLE hFindFile;
		CString strSamplePath; //某个训练样本文件的存取路径

		hFindFile = ::FindFirstFile(strToFind, &findData);

		if (hFindFile != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (findData.cFileName[0] == '.')
					continue;

				if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					strSamplePath = m_vecDirs[i]; //取得类别目录
					strSamplePath += findData.cFileName; //取得完整存取路径
					vecFiles.push_back(strSamplePath); //添加到vecFiles
				}

			} while (::FindNextFile(hFindFile, &findData));

			::FindClose(hFindFile);
		}
		else
		{
			AfxMessageBox(_T("没有找到训练样本图像文件，请检查训练样本目录是否正确!"));
			return false;
		}

		if (vecFiles.size() == 0)// 类别目录为空
			return false;

		m_vecSamples.push_back(vecFiles); //加入该类训练文件的存取路径向量到全部训练样本的向量
	}
	return true;
}


bool CNeuralData::GetTrainingSet()
{
//*
	//清空输入样本矩阵和输出向量矩阵
	m_SetIn.clear();
	m_SetOut.clear();

	vector<double> vecInputs(m_nFeatureDimensions, 0); //定义一个输入样本向量

	int nClass = m_vecDirs.size(); //取得类别数目


	// 读入每一类的训练样本图像，转化为输入向量的形式；设定对应的类标签并编码为输出向量
	for (int i = 0; i<nClass; i++)
	{
		int nSplInClass = m_vecSamples[i].size(); //该类样本数目

												  // 为第 i 类训练样本设定输出向量
		vector<double> outputs(nClass, 0.1);
		outputs[i] = 0.9;

		for (int j = 0; j < nSplInClass; j++)
		{
			// 设定第i类第j个样本的输出向量
			m_SetOut.push_back(outputs);

			String Imgpaths = CT2A(m_vecSamples[i][j]);    //str=CT2A(CStr)实现从CString到string类型转换，CString cstr(str.c_str());实现从string到CString类型转换
		   // 设定第i类第j个样本的输入向量
			Mat ocrImg = imread(Imgpaths, IMREAD_GRAYSCALE); //OCR图像处理对象
			
			int nDim = 0; //输入向量的当前维

			//样本尺寸校验
			if ((ocrImg.rows != m_rt.height) || (ocrImg.cols != m_rt.width))
			{
				AfxMessageBox(_T("图像大小与预设定值不符!请重新设定DigitRec.h中的rows和cols。"));
				return false;
			}

			//图像重采样并按行存储为向量
			if (ocrImg.isContinuous())
			{
				uchar* data = ocrImg.ptr<uchar>(0);
				for (int i = 0; i < m_rt.width*m_rt.height; i += RESAMPLE_LEN)
				{
					vecInputs[nDim] = double(*(data + i));
					nDim++;
				}
			}
			else if (!ocrImg.isContinuous())
			{
				for (int i = 0; i < m_rt.height; i += RESAMPLE_LEN)
				{
					uchar* data = ocrImg.ptr<uchar>(i);
					for (int j = 0; j <m_rt.width; j += RESAMPLE_LEN)
					{
						vecInputs[nDim] = data[j];
						   nDim++;
					}
				}
			}
			 // 设定第i类第j个样本的输入向量			
			m_SetIn.push_back(vecInputs);

		}// for j

	} // for i

	return true;
}
