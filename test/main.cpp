///**
//*Copyright (c) 2018 Young Fan.All Right Reserved.
//*Author: Young Fan
//*Date: 2018.6.23
//*IDE: Visual Studio 2017
//*Description: OpenCV调用摄像头，使用的工具是：IP摄像头app
//*/
//
//#include<opencv2/opencv.hpp>
//#include<highgui/highgui.hpp>
//
//#include<iostream>
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	VideoCapture capture;
//	Mat frame;
//
//	//注意下面的连接部分，admin:123（账号密码），
//	//@符号之后的是局域网ip地址(打开app后，点击下方“打开IP摄像头服务器”，会有显示局域网ip)
//	//即：http://<USERNAME>:<PASSWORD>@<IP_ADDRESS>/<the value of src>
//	capture.open("http://admin:admin@192.168.199.171:8081");
//	//capture.open("rtsp://admin:admin@192.168.199.158:8554/live");
//	if (!capture.isOpened()) { //判断能够打开摄像头
//		cout << "can not open the camera" << endl;
//		cin.get();
//		exit(1);
//	}
//
//	//循环显示每一帧
//	while (1)
//	{
//		capture >> frame;//读取当前每一帧画面
//		imshow("读取视频", frame);//显示当前帧
//		waitKey(30);//延时30ms
//	}
//
//	return 0;
//}

//OpenCV2
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/highgui/highgui.hpp>

//C++11多线程
#include <thread>
#include <iostream>
#include <mutex>
#include<time.h>  

//OpenCV3
#include < stdio.h >  
#include < opencv2\opencv.hpp >  
#include < opencv2\stitching.hpp >  

using namespace cv;
using namespace std;

enum ReturnStatus {
	success = 0,
	failure = 1
};

enum Priority {
	Priority_first = 0,
	Priority_second = 1
};

//Buffer状态
enum HostBufferStatus {
	Empty = 0,
	Buffer0Full = 1,
	Buffer1Full = 2,
	Full = 3
};

/**将uchar类型的数据转换为Mat类型*/
void ucharToMat(uchar *p2, Mat& src)
{
	int nr = src.rows;
	int nc = src.cols * src.channels();//每一行的元素个数
	for (int j = 0; j < nr; j++)
	{
		uchar* data = src.ptr<uchar>(j);
		for (int i = 0; i < nc; i++)
		{
			*data++ = *p2++;
		}
	}
}

/** @function readme */
void readme()
{
	std::cout << " Usage: Panorama < img1 > < img2 >" << std::endl;
}

//帧缓冲区模型示意图
////////////////////////////////
//   -------           --------
//  |  Mat* |         | uchar* |
//   -------           --------
//  |  Mat* |         | uchar* |
//   -------   --->    --------
//  |  Mat* |         | uchar* |
//   -------           --------
//  |  Mat* |         | uchar* |
//   -------           --------
//     Host             Device
////////////////////////////////
typedef struct Frame {
	unsigned char  *data;
	size_t  size;  // data size
	int FrameRows;
	int FrameCols;
	long  index;
} Frame;

class FrameBuffer
{
public:
	//缓冲区尺寸,GPU使用标志位,帧类型
	FrameBuffer(int FrameBufferSize, bool IsUseGPU, int rows, int cols, int type);
	~FrameBuffer();
private:
	std::mutex data_lock;
	//缓冲区尺寸
	int BufferSize = 10;
	//帧类型
	int FrameRows;
	int FrameCols;
	int FrameType = 3;//1:灰度图，3:彩图
	//GPU使用标志位
	bool CudaUsed = false;
	//抽取帧
	Mat Frame;
	//Buffer编号
	int BufferIndex = 0;
	//Frame偏移编号
	int BufferFrameIndex = 0;
	//Buffer装满标志
	bool HostFramebufferFull[2] = {false, false};
	bool DevFramebufferFull = false;
	//Buffer0执行优先级
	int Buffer_priority = Priority_first;
	//输入帧号
	unsigned int in_index = 0;
	//输出帧号
	unsigned int out_index = 0;
	int savecount;
	int copycount = -1;//用于指示两个保存数组，使用哪一个，双缓冲机制
public:
	//输入帧内存双缓冲
	unsigned char* HostInputFramebuffer[2];
	//输入帧显存缓冲
	unsigned char* DeviceInputFramebuffer;
public:
	//成员函数
	ReturnStatus FrameLoad2Buffer(VideoCapture capture);//抽帧载入缓冲
	ReturnStatus HostBufferLoad2DevBuffer();
	unsigned int getHostBufferSize();
	unsigned int getRTFrameIndex();
	unsigned int getRTFrameRows();
	unsigned int getRTFrameCols();
	int getHostFramebufferFull();//0:未满，1：Buffer0满，2：Buffer1满
	void resetHostFramebufferFull();
	bool getDevFramebufferFull();
	void setDevFramebufferFull();
	void resetDevFramebufferFull();
};

FrameBuffer::FrameBuffer(int FrameBufferSize, bool IsUseGPU, int rows, int cols, int type) :BufferSize(FrameBufferSize), CudaUsed(IsUseGPU),
FrameRows(rows), FrameCols(cols), FrameType(type)
{
	cout << "FrameBuffer" << endl;
	in_index = 0;
	out_index = 0;
	savecount = 0;
	copycount = 0;
	for (int i = 0; i < 2; ++i) {
		this->HostInputFramebuffer[i] = (uchar*)malloc(sizeof(uchar) * FrameRows * FrameCols * FrameType * BufferSize);
	}
	if (CudaUsed) {
		size_t avail;
		size_t total;
		cudaMemGetInfo(&avail, &total);
		size_t used = total - avail;
		std::cout << "=================" << std::endl;
		std::cout << "Device memory used: " << used / 1024 /1024 << std::endl;
		std::cout << "Total memory  used: " << total / 1024 / 1024 << std::endl;

		cudaError_t cudaStatus = cudaMalloc((void**)&this->DeviceInputFramebuffer, sizeof(unsigned char) * FrameRows * FrameCols * FrameType * BufferSize);

		cudaMemGetInfo(&avail, &total);
		used = total - avail;
		std::cout << "=================" << std::endl;
		std::cout << "Device memory used: " << used / 1024 / 1024 << std::endl;
		std::cout << "Total memory  used: " << total / 1024 / 1024 << std::endl;

		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
			system("pause");
		}
		else {

		}
	}
}

FrameBuffer::~FrameBuffer()
{
	cout << "~FrameBuffer" << endl;
	for (int i = 0; i < 2; ++i) {
		free(this->HostInputFramebuffer[i]);
		this->HostInputFramebuffer[i] = nullptr;
	}
	if (CudaUsed) {
		cudaFree(this->DeviceInputFramebuffer);
	}
}

ReturnStatus FrameBuffer::FrameLoad2Buffer(VideoCapture capture) {
	//读取下一帧
	if (!capture.read(this->Frame))
	{
		cout << "读取视频失败" << endl;
		return failure;
	}
	in_index++;//实时帧号
	this->BufferIndex = this->copycount / this->BufferSize;//Buffer号
	this->BufferFrameIndex = this->copycount % this->BufferSize;//Frame号

	//载入主机Buffer
	memcpy(HostInputFramebuffer[BufferIndex] + BufferFrameIndex * this->FrameRows * this->FrameCols * this->FrameType,
		this->Frame.data , this->FrameRows * this->FrameCols * this->FrameType);
	//调试用
	//if (in_index == 250) {
	//	Mat t(this->getRTFrameCols(), this->getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
	//	ucharToMat(HostInputFramebuffer[BufferIndex], t);
	//}

	//缓冲区已填满
	//data_lock.lock();
	std::thread::id threadid = std::this_thread::get_id();
	if (0 == this->BufferIndex && (this->BufferFrameIndex == this->BufferSize - 1)) {
		this->HostFramebufferFull[0] = true;
		if (this->HostFramebufferFull[1]) {
			Buffer_priority = Priority_second;//HostFramebuffer[0]后拷贝
		}
		else {
			Buffer_priority = Priority_first;//HostFramebuffer[0]先拷贝
		}
		//若使用GPU，则载入显存
		ReturnStatus Load2Dev;
		if (CudaUsed) {
			//time_t start = clock();
			Load2Dev = this->HostBufferLoad2DevBuffer();
			if (Load2Dev != success) {
				printf("threadid : %d, HostBufferLoad2DevBuffer Error \n", threadid);
			}
			//time_t end = clock();
			//double totaltime = (double)(end - start);
			//printf("in_index : %d, totaltime : %f\n", in_index, totaltime);
			//Mat frame1(this->getRTFrameCols(), this->getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
			//unsigned char* host1 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3);
			//for (int i = 0; i < 50;++i) {
			//	cudaMemcpy(host1, this->DeviceInputFramebuffer + i * sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3, sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
			//	ucharToMat(host1, frame1);
			//	imshow("in_index", frame1);
			//	waitKey(10);
			//}
		}
	}
	else if (1 == this->BufferIndex && (this->BufferFrameIndex == this->BufferSize - 1)) {
		this->HostFramebufferFull[1] = true;
		if (this->HostFramebufferFull[0]) {
			this->Buffer_priority = Priority_first;//HostFramebuffer[0]先拷贝
		}
		else {
			this->Buffer_priority = Priority_second;//HostFramebuffer[0]后拷贝
		}
		//若使用GPU，则载入显存
		ReturnStatus Load2Dev;
		if (CudaUsed) {
			Load2Dev = this->HostBufferLoad2DevBuffer();
			if (Load2Dev != success) {
				printf("threadid : %d, HostBufferLoad2DevBuffer Error \n", threadid);
			}
			//Mat frame1(this->getRTFrameCols(), this->getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
			//unsigned char* host1 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3);
			//for (int i = 0; i < 50; ++i) {
			//	cudaMemcpy(host1, this->DeviceInputFramebuffer + i * sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3, sizeof(unsigned char) * this->getRTFrameRows() * this->getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
			//	ucharToMat(host1, frame1);
			//	imshow("in_index", frame1);
			//	waitKey(10);
			//}
		}
	}
	//data_lock.unlock();

	this->copycount++;
	if (this->copycount == 2 * BufferSize) {
		this->copycount = 0;//偏移计数到上限，则清零
	}
	return success;
}

ReturnStatus FrameBuffer::HostBufferLoad2DevBuffer() {
	if (!this->DevFramebufferFull && this->HostFramebufferFull[0] && this->HostFramebufferFull[1]) {
		if (this->Buffer_priority == Priority_first) {
			cudaError_t status = cudaMemcpy(this->DeviceInputFramebuffer, this->HostInputFramebuffer[0], this->BufferSize * sizeof(unsigned char) * this->FrameRows * this->FrameCols * this->FrameType, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CudaMemcpyHostToDevice Error \n");
			}
			this->DevFramebufferFull = true;//拷贝完成，DevBuffer填满
			this->HostFramebufferFull[0] = false;//拷贝完成，HostBuffer清空
			return cudaSuccess == status ? success : failure;
		}
		else {
			cudaError_t status = cudaMemcpy(this->DeviceInputFramebuffer, this->HostInputFramebuffer[1], this->BufferSize * sizeof(unsigned char) * this->FrameRows * this->FrameCols * this->FrameType, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CudaMemcpyHostToDevice Error \n");
			}
			this->DevFramebufferFull = true;//拷贝完成，DevBuffer填满
			this->HostFramebufferFull[1] = false;//拷贝完成，HostBuffer清空
			return cudaSuccess == status ? success : failure;
		}
	}
	else if (!this->DevFramebufferFull && this->HostFramebufferFull[0]) {
		cudaError_t status = cudaMemcpy(this->DeviceInputFramebuffer, this->HostInputFramebuffer[0], this->BufferSize * sizeof(unsigned char) * this->FrameRows * this->FrameCols * this->FrameType, cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			fprintf(stderr, "CudaMemcpyHostToDevice Error \n");
		}
		this->DevFramebufferFull = true;//拷贝完成，DevBuffer填满
		this->HostFramebufferFull[0] = false;//拷贝完成，HostBuffer清空
		return cudaSuccess == status ? success : failure;
	}
	else if (!this->DevFramebufferFull && this->HostFramebufferFull[1]) {
		cudaError_t status = cudaMemcpy(this->DeviceInputFramebuffer, this->HostInputFramebuffer[1], this->BufferSize * sizeof(unsigned char) * this->FrameRows * this->FrameCols * this->FrameType, cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			fprintf(stderr, "CudaMemcpyHostToDevice Error \n");
		}
		this->DevFramebufferFull = true;//拷贝完成，DevBuffer填满
		this->HostFramebufferFull[1] = false;//拷贝完成，HostBuffer清空
		return cudaSuccess == status ? success : failure;
	}
	else {
		return failure;
	}
}

unsigned int FrameBuffer::getHostBufferSize() {
	return this->BufferSize;
}

unsigned int FrameBuffer::getRTFrameIndex() {
	return this->in_index;
}

unsigned int FrameBuffer::getRTFrameRows() {
	return this->FrameRows;
}

unsigned int FrameBuffer::getRTFrameCols() {
	return this->FrameCols;
}

int FrameBuffer::getHostFramebufferFull() {
	if (this->HostFramebufferFull[0] && this->HostFramebufferFull[1]) {
		return 3;
	}
	else if (this->HostFramebufferFull[0]) {
		return 1;
	}
	else if (this->HostFramebufferFull[1]) {
		return 2;
	}
	else {
		return 0;
	}
}

void FrameBuffer::resetHostFramebufferFull() {
	this->HostFramebufferFull[0] = false;
	this->HostFramebufferFull[1] = false;
}

bool FrameBuffer::getDevFramebufferFull() {
	return this->DevFramebufferFull;
}

void FrameBuffer::setDevFramebufferFull() {
	this->DevFramebufferFull = true;
}

void FrameBuffer::resetDevFramebufferFull() {
	this->DevFramebufferFull = false;
}

//displaythread
void display(const char *name) {
	VideoCapture capture(name);
	if (!capture.isOpened()) { //判断能够打开摄像头
		cout << "can not open the camera" << endl;
		exit(1);
	}
	Mat frame;
	namedWindow(name);
	while (1) {
		if (!capture.read(frame))
		{
			cout << "读取视频失败" << endl;
			break;
		}
		imshow(name, frame);
		waitKey(10);
	}
}

//writethread
void VedioLoad(FrameBuffer &FrameBuffer, const char *name) {
	VideoCapture capture(name);
	if (!capture.isOpened()) { //判断能够打开摄像头
		cout << "can not open the camera" << endl;
		exit(1);
	}
	std::thread::id threadid = std::this_thread::get_id();
	ReturnStatus LoadStatus;
	while (1)
	{
		ReturnStatus LoadStatus = FrameBuffer.FrameLoad2Buffer(capture);
		if (LoadStatus == failure) {
			cout << "vedio stasus: " << capture.get(2) << endl;
			break;
		}
		cv::waitKey(20);//延时20ms
		//cout << "vedioname: " << name << " ,Frame:" << capture.get(7) << endl;
		//cout << "threadid: " << threadid << " ,FrameIndex:" << FrameBuffer.getRTFrameIndex() << endl;
	}
	return;
}

//readthread
void FrameStitch(FrameBuffer &FrameBuffer1, FrameBuffer &FrameBuffer2) {
	unsigned int count = 1;//帧计数
	while (1)
	{
		//if (FrameBuffer1.getDevFramebufferFull()) {
		//	//调试用
		//	Mat t(FrameBuffer1.getRTFrameCols(), FrameBuffer1.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
		//	ucharToMat(FrameBuffer1.HostInputFramebuffer[0], t);
		//	Mat frame1(FrameBuffer1.getRTFrameCols(), FrameBuffer1.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
		//	unsigned char* host1 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3);
		//	cudaMemcpy(host1, FrameBuffer1.DeviceInputFramebuffer, sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
		//	ucharToMat(host1, frame1);
		//}

		//if (FrameBuffer2.getDevFramebufferFull()) {
		//	//调试用
		//	Mat t(FrameBuffer2.getRTFrameCols(), FrameBuffer2.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
		//	ucharToMat(FrameBuffer2.HostInputFramebuffer[0], t);
		//	Mat frame1(FrameBuffer2.getRTFrameCols(), FrameBuffer2.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
		//	unsigned char* host1 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * FrameBuffer2.getRTFrameRows() * FrameBuffer2.getRTFrameCols() * 3);
		//	cudaMemcpy(host1, FrameBuffer2.DeviceInputFramebuffer, sizeof(unsigned char) * FrameBuffer2.getRTFrameRows() * FrameBuffer2.getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
		//	ucharToMat(host1, frame1);
		//}
			
		if (FrameBuffer1.getDevFramebufferFull() && FrameBuffer2.getDevFramebufferFull()) {
			//Mat frame1(FrameBuffer1.getRTFrameCols(), FrameBuffer1.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
			//Mat frame2(FrameBuffer2.getRTFrameCols(), FrameBuffer2.getRTFrameRows(), CV_8UC3, Scalar(0, 0, 0));
			//unsigned char* host1 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3);
			//unsigned char* host2 = (uchar*)malloc(sizeof(uchar) * sizeof(unsigned char) * FrameBuffer2.getRTFrameRows() * FrameBuffer2.getRTFrameCols() * 3);
			for (int i = 0; i < FrameBuffer1.getHostBufferSize(); ++i) {
				//cudaError_t cudaStatus1 = cudaMemcpy(host1, FrameBuffer1.DeviceInputFramebuffer + i * sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3, sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
				//cudaError_t cudaStatus2 = cudaMemcpy(host2, FrameBuffer2.DeviceInputFramebuffer + i * sizeof(unsigned char) * FrameBuffer1.getRTFrameRows() * FrameBuffer1.getRTFrameCols() * 3, sizeof(unsigned char) * FrameBuffer2.getRTFrameRows() * FrameBuffer2.getRTFrameCols() * 3, cudaMemcpyDeviceToHost);
				//if (cudaStatus1 != cudaSuccess)
				//{
				//	fprintf(stderr, "cudaStatus1: %s\n", cudaGetErrorString(cudaStatus1));
				//}
				//if (cudaStatus2 != cudaSuccess)
				//{
				//	fprintf(stderr, "cudaStatus2: %s\n", cudaGetErrorString(cudaStatus2));
				//}
				//ucharToMat(host1, frame1);
				//ucharToMat(host2, frame2);
				//namedWindow("frame1", WINDOW_NORMAL);
				//imshow("frame1", frame1);
				//namedWindow("frame2", WINDOW_NORMAL);
				//imshow("frame2", frame2);
				printf("count: %d\n", count);
				//waitKey(10);
				count++;
				if (count == FrameBuffer1.getHostBufferSize()) {
					FrameBuffer1.resetDevFramebufferFull();
					FrameBuffer2.resetDevFramebufferFull();
					printf("FrameBuffer1 DevFramebufferFull flag clear, FrameBuffer2 DevFramebufferFull flag clear\n");
					count = 1;
				}
			}
		}
	}
	return;
}

int main()
{
	VideoCapture capture;
	capture.open("video_left.mp4");
	VideoCapture capture1;
	capture1.open("video_right.mp4");
	FrameBuffer FrameBufferThread1(50, true, 1920, 1080, 3);
	FrameBuffer FrameBufferThread2(50, true, 1920, 1080, 3);
	std::thread thread1(VedioLoad, std::ref(FrameBufferThread1), "video_left.mp4");
	std::thread thread2(VedioLoad, std::ref(FrameBufferThread2), "video_right.mp4");
	std::thread thread3(FrameStitch, std::ref(FrameBufferThread1), std::ref(FrameBufferThread2));
	//std::thread thread1(display, "video_left.mp4");
	//std::thread thread2(display, "video_right.mp4");
	thread1.join();
	thread2.join();
	thread3.join();
	std::cout << "主线程运行结束" << std::endl;
	return 0;


	//VideoCapture capture;
	//Mat frame;

	//注意下面的连接部分，admin:123（账号密码），
	//@符号之后的是局域网ip地址(打开app后，点击下方“打开IP摄像头服务器”，会有显示局域网ip)
	//即：http://<USERNAME>:<PASSWORD>@<IP_ADDRESS>/<the value of src>
	//capture.open("http://admin:admin@192.168.199.171:8081");
	//capture.open("rtsp://admin:admin@192.168.199.158:8554/live");
	//capture.open("video_left.mp4");
	//if (!capture.isOpened()) { //判断能够打开摄像头
	//	cout << "can not open the camera" << endl;
	//	exit(1);
	//}

	////循环显示每一帧
	//int FPS = capture.get(CAP_PROP_FPS);
	//FrameBuffer FrameBufferThread(50, true, capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT), 3);
	//while (1)
	//{
	//	FrameBufferThread.FrameLoadBuffer(capture);
	//	//waitKey(30);//延时30ms
	//	Mat tempFrame(capture.get(CAP_PROP_FRAME_HEIGHT), capture.get(CAP_PROP_FRAME_WIDTH), CV_8UC3, Scalar(0, 0, 0));
	//	if (FrameBufferThread.getRTFrameIndex() == 40)
	//		ucharToMat(FrameBufferThread.HostInputFramebuffer[0], tempFrame);
	//	else if(FrameBufferThread.getRTFrameIndex() == 80)
	//		ucharToMat(FrameBufferThread.HostInputFramebuffer[1], tempFrame);
	//}

	//std::vector< cv::Mat > vImg;
	//cv::Mat rImg;

	//vImg.push_back(cv::imread("left.jpg"));
	//vImg.push_back(cv::imread("right.jpg"));

	//cv::Stitcher stitcher = cv::Stitcher::createDefault();

	//unsigned long AAtime = 0, BBtime = 0; //check processing time
	//AAtime = cv::getTickCount(); //check processing time

	//cv::Stitcher::Status status = stitcher.stitch(vImg, rImg);

	//BBtime = cv::getTickCount(); //check processing time 
	//printf("Time consuming: %.2lf sec \n", (BBtime - AAtime) / cv::getTickFrequency()); //check processing time

	//cv::namedWindow("Stitching Result", WINDOW_NORMAL);
	//if (cv::Stitcher::OK == status)
	//	cv::imshow("Stitching Result", rImg);
	//else
	//	printf("Stitching fail.");

	//cv::waitKey(0);
}