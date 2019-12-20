#include <iostream>
#include <vector>
#include <time.h>

// opencv头文件
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

static bool is_camera = false, is_view = false, is_save = false, is_debug = false;
static int cam_width, cam_height, cam_num;

int main(int argc, char* argv[])
{
	//	输入视频流
	vector<VideoCapture> captures;
	string video_IPaddress[2];;
	video_IPaddress[0] = "http://admin:admin@192.168.199.171:8081";
	video_IPaddress[1] = "http://admin:admin@192.168.199.240:8081";
	vector<string> video_names{ "video_left.mp4", "video_right.mp4" };
	if (is_camera)
	{
		for (int cam_idx = 0; cam_idx < cam_num; cam_idx++)
		{
			VideoCapture cam_cap;
			if (cam_cap.open(cam_idx))
			{
				cam_cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
				cam_cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
				cam_cap.set(CV_CAP_PROP_FPS, 15);
				captures.push_back(cam_cap);
				cout << "camera " << cam_idx << " opened successfully." << endl;
			}
			else if (cam_cap.open(video_IPaddress[cam_idx])) {
				captures.push_back(cam_cap);
				cout << "camera " << cam_idx << " opened successfully." << endl;
			}
			else
				break;
		}
		if (captures.size() == 0)
		{
			cout << "No camera captured. Please check!" << endl;
			return -1;
		}
	}
	else
	{
		int video_num = video_names.size();
		captures.resize(video_num);
		for (int i = 0; i < video_num; i++)
		{
			captures[i].open(video_names[i]);
			if (!captures[i].isOpened())
			{
				cout << "Fail to open " << video_names[i] << endl;
				for (int j = 0; j < i; j++) captures[j].release();
				return -1;
			}
		}
	}
	cout << "Video capture success" << endl;
	
	// 创建结果视频
	VideoWriter writer;
	bool firstframe = true, is_preview_ = true;
	long frame_time = 0;
	double fps = captures[0].get(CV_CAP_PROP_FPS);
	int frame_show_interval = cvFloor(1000 / fps);
	double show_scale = 1.0, scale_interval = 0.03;
	string window_name("视频拼接");

	int FrameIndex = 0;
	Mat Frameleft, Frameright;
	Mat rImg, show_dst;
	unsigned long AAtime = 0, BBtime = 0; //check processing time
	AAtime = cv::getTickCount(); //check processing time
	while (captures[0].read(Frameleft) && captures[1].read(Frameright)) {
		vector< Mat > Frames{ Frameleft , Frameright };
		cv::Stitcher stitcher = cv::Stitcher::createDefault();
		cv::Stitcher::Status status = stitcher.stitch(Frames, rImg);
		if (firstframe) {
			writer.open("vedio_save.avi", CV_FOURCC('D', 'I', 'V', '3'), 20, Size(rImg.cols, rImg.rows));
			firstframe = false;
		}
		long write_start_clock = clock();
		writer.write(rImg);
		long write_clock = clock();
		cout << write_clock - write_start_clock << "ms)";
		frame_time += write_clock - write_start_clock;
		cout << "当前帧号： "  << FrameIndex++ << endl;
		//	显示---
		if (is_preview_)
		{
			int key = waitKey(std::max(1, (int)(frame_show_interval - frame_time)));
			if (key == 27)	//	ESC
				break;
			else if (key == 61 || key == 43)	//	+
				show_scale += scale_interval;
			else if (key == 45)				//	-
				if (show_scale >= scale_interval)
					show_scale -= scale_interval;
			resize(rImg, show_dst, Size(show_scale * rImg.cols, show_scale * rImg.rows));
			namedWindow(window_name, WINDOW_NORMAL);
			imshow(window_name, show_dst);
		}
	}
	BBtime = cv::getTickCount(); //check processing time 
	printf("Time consuming: %.2lf sec \n", (BBtime - AAtime) / cv::getTickFrequency()); //check processing time

	writer.release();
	return 0;
}
