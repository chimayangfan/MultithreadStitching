#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/cudaobjdetect.hpp>

using namespace std;
using namespace cv;
using namespace cuda;
#ifdef _DEBUG
#pragma comment ( lib,"opencv_core340d.lib")
#pragma comment ( lib,"opencv_highgui340d.lib")
#pragma comment ( lib,"opencv_calib3d340d.lib")
#pragma comment ( lib,"opencv_imgcodecs340d.lib")
#pragma comment ( lib,"opencv_imgproc340d.lib")
#pragma comment ( lib,"opencv_cudaimgproc340d.lib")
#pragma comment ( lib,"opencv_cudaarithm340d.lib")
#pragma comment ( lib,"cudart.lib")
#else
#pragma comment ( lib,"opencv_core340.lib")
#pragma comment ( lib,"opencv_highgui340.lib")
#pragma comment ( lib,"opencv_calib3d340.lib")
#pragma comment ( lib,"opencv_imgcodecs340.lib")
#pragma comment ( lib,"opencv_imgproc340.lib")
#pragma comment ( lib,"opencv_cudaimgproc340.lib")
#pragma comment ( lib,"opencv_cudaarithm340.lib")
#pragma comment ( lib,"cudart.lib")
#endif

//出错处理函数
#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}



int main(int argc, char **argv)
{

	VideoCapture cap;
	//cap.open("rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0");
	cap.open(0);
	if (!cap.isOpened())
		return -1;

	Mat frame;
	cuda::GpuMat cu_dst;

	Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create("D:\\opencv3_4_0\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml");

	vector<Rect> faces;

	while (1)
	{
		cap >> frame;
		if (frame.empty())
			break;

		cuda::GpuMat image_gpu(frame);

		cascade_gpu->detectMultiScale(image_gpu, cu_dst);

		cascade_gpu->convert(cu_dst, faces);

		for (int i = 0; i < faces.size(); ++i)
			rectangle(frame, faces[i], Scalar(255));

		imshow("faces", frame);
		//等待用户按键
		waitKey(1);

	};

	cap.release();
	return 0;
}
