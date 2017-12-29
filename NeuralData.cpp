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
	Clear();                                //��ո�������
	m_rt = SrcRect;
	m_nFeatureDimensions = FeatureNumber;   //�趨������Ŀ
}

/*
GetClassInfoFromDir ÿ��ͬ�����ѵ����������ͬһ�����������Ϊ�����ļ����У����в�ͬ�����������ļ����ַ���һ����ַΪstrDir�����ļ����У�
                    �������ļ��������ļ���������Ϣ���ѵ����������Ŀ¼�����������Ϣ,���������
strDir���������ѵ�����������ļ��е�ַ
*/
void CNeuralData::GetClassInfoFromDir(CString strDir)  
{
	if (strDir.Right(1) != "\\")
		strDir = strDir + "\\";

	CString strDirName; //����Ŀ¼���ƣ��������

						//����ѵ�������ļ����µķ���Ŀ¼
	WIN32_FIND_DATA findData;
	HANDLE hFindFile;

	CString strDirFind = strDir;
	strDirFind += "*.*";
	hFindFile = ::FindFirstFile(strDirFind, &findData);

	Clear(); //���Ŀ¼����

	if (hFindFile != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (findData.cFileName[0] == '.')  //..��ʾ��ǰĿ¼���ϼ�Ŀ¼�����ж��ļ�������ĸ�ǲ���"."��
				continue;
			if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))//�ж��Ƿ�Ϊ�ļ��У�Ŀ���ļ���ǣ��ļ��б��16�����������ļ���ʱ��������ѭ��
				continue;

			strDirFind = strDir + findData.cFileName;

			strDirFind += "\\";

			strDirName = findData.cFileName;
			AddData(strDirFind, strDirName);

		} while (::FindNextFile(hFindFile, &findData));//��ǰĿ¼�Ƿ񻹴��ڡ���һ��Ŀ¼�����ߡ��ļ���

		::FindClose(hFindFile);
	}
	else
	{
		AfxMessageBox(_T("û���ҵ�ѵ������Ŀ¼���Ӷ��޷���ȡ���������Ϣ!"));
		return;
	}
}

/*
AddData��������·����Ϣ���ļ��й�����Ϣ�����Ҫʶ�����𣬲��ۼ��������
strImgDir����ʾĳһ������ͼƬ���ļ��е�ַ
strClassName����ʾĳһ���������������
*/
bool CNeuralData::AddData(CString strImgDir, CString strClassName)
{
	//��������ļ�����Ŀ¼��·����Ϣ
	m_vecDirs.push_back(strImgDir);

	//��Ӵ������������֣������Դ�Ÿ�����ļ��е�������Ϊ�������
	m_vecClassNames.push_back(strClassName);

	m_nClassNumber++; //�����Ŀ
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
	//ȡ��ѵ�������Ĵ�ȡ·��
	if (GetSamplePaths() == false)
		return false;

	//ȡ������ѵ����
	if (!GetTrainingSet())
		return false;

	return true;
}

bool CNeuralData::GetSamplePaths()
{
	int nClass = m_vecDirs.size(); //�����Ŀ

	for (int i = 0; i<nClass; i++)
	{
		// ������i�����Ŀ¼������Щ�����ļ��Ĵ�ȡ·�������� vecFiles
		vector<CString> vecFiles; //ĳһ��ѵ�������Ĵ�ȡ·��

								  // ����Ŀ��Ϊ��i�����Ŀ¼�µ�ȫ�� .bmp ͼ���ļ�
		CString strToFind = m_vecDirs[i];
		strToFind += "*.bmp";

		WIN32_FIND_DATA findData;
		HANDLE hFindFile;
		CString strSamplePath; //ĳ��ѵ�������ļ��Ĵ�ȡ·��

		hFindFile = ::FindFirstFile(strToFind, &findData);

		if (hFindFile != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (findData.cFileName[0] == '.')
					continue;

				if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					strSamplePath = m_vecDirs[i]; //ȡ�����Ŀ¼
					strSamplePath += findData.cFileName; //ȡ��������ȡ·��
					vecFiles.push_back(strSamplePath); //��ӵ�vecFiles
				}

			} while (::FindNextFile(hFindFile, &findData));

			::FindClose(hFindFile);
		}
		else
		{
			AfxMessageBox(_T("û���ҵ�ѵ������ͼ���ļ�������ѵ������Ŀ¼�Ƿ���ȷ!"));
			return false;
		}

		if (vecFiles.size() == 0)// ���Ŀ¼Ϊ��
			return false;

		m_vecSamples.push_back(vecFiles); //�������ѵ���ļ��Ĵ�ȡ·��������ȫ��ѵ������������
	}
	return true;
}


bool CNeuralData::GetTrainingSet()
{
//*
	//���������������������������
	m_SetIn.clear();
	m_SetOut.clear();

	vector<double> vecInputs(m_nFeatureDimensions, 0); //����һ��������������

	int nClass = m_vecDirs.size(); //ȡ�������Ŀ


	// ����ÿһ���ѵ������ͼ��ת��Ϊ������������ʽ���趨��Ӧ�����ǩ������Ϊ�������
	for (int i = 0; i<nClass; i++)
	{
		int nSplInClass = m_vecSamples[i].size(); //����������Ŀ

												  // Ϊ�� i ��ѵ�������趨�������
		vector<double> outputs(nClass, 0.1);
		outputs[i] = 0.9;

		for (int j = 0; j < nSplInClass; j++)
		{
			// �趨��i���j���������������
			m_SetOut.push_back(outputs);

			String Imgpaths = CT2A(m_vecSamples[i][j]);    //str=CT2A(CStr)ʵ�ִ�CString��string����ת����CString cstr(str.c_str());ʵ�ִ�string��CString����ת��
		   // �趨��i���j����������������
			Mat ocrImg = imread(Imgpaths, IMREAD_GRAYSCALE); //OCRͼ�������
			
			int nDim = 0; //���������ĵ�ǰά

			//�����ߴ�У��
			if ((ocrImg.rows != m_rt.height) || (ocrImg.cols != m_rt.width))
			{
				AfxMessageBox(_T("ͼ���С��Ԥ�趨ֵ����!�������趨DigitRec.h�е�rows��cols��"));
				return false;
			}

			//ͼ���ز��������д洢Ϊ����
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
			 // �趨��i���j����������������			
			m_SetIn.push_back(vecInputs);

		}// for j

	} // for i

	return true;
}
