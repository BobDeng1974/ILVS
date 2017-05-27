#include "common.hpp"

using namespace cv;
using namespace std;


Mat images[NumberOfCamera]; //Global camera frame, can access everywhere but need to use mutex lock

pthread_mutex_t cam_mutex[3] = {
		PTHREAD_MUTEX_INITIALIZER,
		PTHREAD_MUTEX_INITIALIZER,
		PTHREAD_MUTEX_INITIALIZER
};


void *camera_capture(void *arg) {
	int cam = *((int *) arg);
	Mat temp_frame;
	cout << "in2" << endl;
	VideoCapture cap("../../media/lb3_10.mov");
	if(!cap.isOpened()){
		cout << "can't open device" << endl;
	    return (void*)-1;
	}
	cout << "in3" << endl;
	while(true){
	    if(!cap.read(temp_frame))
	        break;

	    pthread_mutex_lock(&cam_mutex[cam]);
	    images[cam] = cv::Mat(temp_frame);
	    pthread_mutex_unlock(&cam_mutex[cam]);

	    imshow( "window",  images[cam] );
	    if(waitKey(1) >= 0) break;
	}
	return 0;
}

/*
int main( int argc, char** argv )
{

	pthread_t thread_1,thread_2, thread_3;
	int cam = 0;

	int cam_1 = pthread_create(&thread_1, NULL, camera_capture, (void *)&cam);
//	int cam_2 = pthread_create(&thread_2, NULL, camera_capture, (void *)1);
//	int cam_3 = pthread_create(&thread_3, NULL, camera_capture, (void *)2);

	while(1){

	    if(waitKey(1) >= 0) break;
	};
	return 0;
}

*/
