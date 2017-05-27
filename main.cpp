#include "common.hpp"
#include "shader.hpp"

#define LOGLN(msg) std::cout << msg << std::endl
#define GLOffScreen false
#define CVOffScreen true
#define WindowWidth 1280
#define WindowHeight 720
#define Division 20
#define INPUT1 "../../media/lb2_10.mov"
#define INPUT2 "../../media/lb3_10.mov"
#define INPUT3 "../../media/lb4_10.mov"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static vector<GLuint> tex;
static vector<Mat> images;
static Mat pano_out;
vector<VideoCapture> cap;

static GLFWwindow* window;
static GLuint vertexbuffer;
static GLuint uvbuffer;
static GLuint sectionbuffer;
static GLuint programID;

static GLuint TextureID;
static GLuint vertexPosition_modelspaceID;
static GLuint vertexUVID;
static GLuint sectionID;

static int vertexBuf_index;
static int imageEmpty_count;

void gl_init(GLsizei width, GLsizei height){
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf( stderr, "Failed to initialize GLFW\n");
		exit(0);
	}
#if GLOffScreen
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#endif
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	window = glfwCreateWindow(width, height, "GL texture mapping", NULL,
			NULL);
	if (window == NULL) {
		fprintf( stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		exit(0);
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(0);
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//	// Disable depth buffer, Dither, SmoothShading for performance
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_DITHER);
	glShadeModel(GL_FLAT);

	// Enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	programID = LoadShaders("../shader.vert", "../shader.frag");
		// Get a handle for our buffers
	vertexPosition_modelspaceID = glGetAttribLocation(programID, "Position");
	vertexUVID = glGetAttribLocation(programID, "TexCoord");
	TextureID = glGetUniformLocation(programID, "gSampler");
	sectionID = glGetAttribLocation(programID, "Section");
}

int gl_breakVertice(GLfloat g_vertex_buffer_data[], vector<Point2f> new_coord_onPlane, GLfloat division, int startIndex, bool isVertex , GLfloat z_value) {
	GLfloat re_division = 1.0f / division;
	GLfloat slopeAC=1;
	GLfloat slopeBD=1;
	int bufferIndex = startIndex;

	if((new_coord_onPlane[0].x - new_coord_onPlane[3].x) != 0)
		slopeAC = (new_coord_onPlane[0].y - new_coord_onPlane[3].y) / (new_coord_onPlane[0].x - new_coord_onPlane[3].x);

	if ((new_coord_onPlane[1].x - new_coord_onPlane[2].x)!=0)
		slopeBD = (new_coord_onPlane[1].y - new_coord_onPlane[2].y) / (new_coord_onPlane[1].x - new_coord_onPlane[2].x);

	GLfloat re_slopeAC = 1.0f / slopeAC;
	GLfloat re_slopeBD = 1.0f / slopeBD;
	Point2f tempA = Point2f(0.0f, 0.0f);
	Point2f tempB = Point2f(0.0f, 0.0f);
	GLfloat tempSlope = 0.0f;
	GLfloat space_x;

	int vertexNumber = (division + 1) * (division + 1);
	vector<Point2f> smallVertices(vertexNumber, Point2f(0.0f, 0.0f));
	//cout << "total vertex: " << vertexNumber << endl;
	for (int row = 0; row < division + 1; row++) {
		tempA.y = ((new_coord_onPlane[0].y - new_coord_onPlane[3].y) * re_division) * row + new_coord_onPlane[3].y;
		tempB.y = ((new_coord_onPlane[1].y - new_coord_onPlane[2].y) * re_division) * row + new_coord_onPlane[2].y;
		if((new_coord_onPlane[0].x - new_coord_onPlane[3].x) != 0)
			tempA.x = (tempA.y - new_coord_onPlane[3].y + slopeAC * new_coord_onPlane[3].x) * re_slopeAC;
		else
			tempA.x = new_coord_onPlane[0].x;
		if((new_coord_onPlane[1].x - new_coord_onPlane[2].x) != 0)
			tempB.x = (tempB.y - new_coord_onPlane[2].y + slopeBD * new_coord_onPlane[2].x) * re_slopeBD;
		else
			tempB.x = new_coord_onPlane[1].x;
		tempSlope = (tempB.y - tempA.y) / (tempB.x - tempA.x);
		space_x = (tempB.x - tempA.x) * re_division;

		for (int col = 0; col < division + 1; col++) {
			//input vertex info
			float x = tempA.x + space_x * col;
			float y = (tempSlope * space_x * col) + tempA.y;
			int index = (int) (row * (division + 1)) + col;

			smallVertices.at(index).x = x;		// x value;
			smallVertices.at(index).y = y;	//y value		//z value
			//cout << index <<": vertex " << col << ": "<< smallVertices.at(index).x << ", " << smallVertices.at(index).y << ", " << endl; //debug usage
		}
	}

	//Pack the vertex into triangles
	int div = (int) (division);
	if (isVertex == true) {
		for (int i = 0; i < vertexNumber; i++) { //count by the number of triangle which 90 degree angle are on bottom right
			if ((i + 1) % (div + 1) == 0)  //s kip when vertex1 is the rightest vertex
				continue;
			//vertex1
			g_vertex_buffer_data[bufferIndex] = smallVertices.at(i).x;
			g_vertex_buffer_data[bufferIndex + 1] = smallVertices.at(i).y;
			g_vertex_buffer_data[bufferIndex + 2] = z_value;
			//vertex2
			g_vertex_buffer_data[bufferIndex + 3] = smallVertices.at(i + 1).x;
			g_vertex_buffer_data[bufferIndex + 4] = smallVertices.at(i + 1).y;
			g_vertex_buffer_data[bufferIndex + 5] = z_value;
			//vertex3
			g_vertex_buffer_data[bufferIndex + 6] = smallVertices.at(i + div + 2).x;
			g_vertex_buffer_data[bufferIndex + 7] = smallVertices.at(i + div + 2).y;
			g_vertex_buffer_data[bufferIndex + 8] = z_value;
			bufferIndex = bufferIndex + 9;
			//cout << "buf index: " << bufferIndex << "tri vertex: " << i << ","	<< i + 1 << "," << i + div + 2 << endl;
			if (i + div + 2 >= vertexNumber - 1)
				break;	//end when vertex3 is the last vertex in the picture
		}
		for (int i = 0; i < vertexNumber; i++) { //count by the number of triangle which 90 degree angle are on top left
			if ((i + 1) % (div + 1) == 0)  //skip when vertex1 is the rightest vertex
				continue;
			//vertex1
			g_vertex_buffer_data[bufferIndex] = smallVertices.at(i).x;
			g_vertex_buffer_data[bufferIndex + 1] = smallVertices.at(i).y;
			g_vertex_buffer_data[bufferIndex + 2] = z_value;
			//vertex2
			g_vertex_buffer_data[bufferIndex + 3] = smallVertices.at(i + div + 1).x;
			g_vertex_buffer_data[bufferIndex + 4] = smallVertices.at(i + div + 1).y;
			g_vertex_buffer_data[bufferIndex + 5] = z_value;
			//vertex3
			g_vertex_buffer_data[bufferIndex + 6] = smallVertices.at(i + div + 2).x;
			g_vertex_buffer_data[bufferIndex + 7] = smallVertices.at(i + div + 2).y;
			g_vertex_buffer_data[bufferIndex + 8] = z_value;
			bufferIndex = bufferIndex + 9;
			//cout << "buf index: " << bufferIndex << "tri vertex: " << i << ","<< i + div + 1 << "," << i + div + 2 << endl;
			if (i + div + 2 >= vertexNumber - 1)
				break;	//end when vertex3 is the last vertex in the picture
		}
	}else{
		for (int i = 0; i < vertexNumber; i++) { //count by the number of triangle which 90 degree angle are on bottom right
			if ((i + 1) % (div + 1) == 0)  //s kip when vertex1 is the rightest vertex
				continue;
			//vertex1
			g_vertex_buffer_data[bufferIndex] = smallVertices.at(i).x;
			g_vertex_buffer_data[bufferIndex + 1] = smallVertices.at(i).y;
			//vertex2
			g_vertex_buffer_data[bufferIndex + 2] = smallVertices.at(i + 1).x;
			g_vertex_buffer_data[bufferIndex + 3] = smallVertices.at(i + 1).y;
			//vertex3
			g_vertex_buffer_data[bufferIndex + 4] = smallVertices.at(i + div + 2).x;
			g_vertex_buffer_data[bufferIndex + 5] = smallVertices.at(i + div + 2).y;
			//			cout << "inside tex buf: "<< bufferIndex << ":::" << g_vertex_buffer_data[bufferIndex] << "," << g_vertex_buffer_data[bufferIndex + 1] << endl;
			bufferIndex = bufferIndex + 6;

			if (i + div + 2 >= vertexNumber - 1)
				break;	//end when vertex3 is the last vertex in the picture
		}

		for (int i = 0; i < vertexNumber; i++) { //count by the number of triangle which 90 degree angle are on top left
			if ((i + 1) % (div + 1) == 0)  //skip when vertex1 is the rightest vertex
				continue;
			//vertex1
			g_vertex_buffer_data[bufferIndex] = smallVertices.at(i).x;
			g_vertex_buffer_data[bufferIndex + 1] = smallVertices.at(i).y;
			//vertex2
			g_vertex_buffer_data[bufferIndex + 2] = smallVertices.at(i + div + 1).x;
			g_vertex_buffer_data[bufferIndex + 3] = smallVertices.at(i + div + 1).y;
			//vertex3
			g_vertex_buffer_data[bufferIndex + 4] = smallVertices.at(i + div + 2).x;
			g_vertex_buffer_data[bufferIndex + 5] = smallVertices.at(i + div + 2).y;
			bufferIndex = bufferIndex + 6;
			//			cout << "buf index: " << bufferIndex << "tri vertex: " << i << "," << i + div + 1 << "," << i + div + 2 << endl;
			if (i + div + 2 >= vertexNumber - 1)
				break;	//end when vertex3 is the last vertex in the picture
		}
	}
	return bufferIndex;
}

void gl_release(){
	//Close the vertex and tex coord array from GPU
	glDisableVertexAttribArray(vertexPosition_modelspaceID);
	glDisableVertexAttribArray(vertexUVID);

	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &TextureID);
	glDeleteTextures(1, &tex[0]);
	glDeleteTextures(1, &tex[1]);
	glDeleteTextures(1, &tex[2]);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

GLuint matToTexture(Mat &mat, GLenum minFilter, GLenum magFilter,
		GLenum wrapFilter, GLuint tex_num) {
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(tex_num, &textureID);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR
			|| magFilter == GL_LINEAR_MIPMAP_NEAREST
			|| magFilter == GL_NEAREST_MIPMAP_LINEAR
			|| magFilter == GL_NEAREST_MIPMAP_NEAREST) {
#if DEBUG
		cout
		<< "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR"
		<< endl;
#endif
		magFilter = GL_LINEAR;
	}

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1) {
		inputColourFormat = GL_LUMINANCE;
	}

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
			0,           // Pyramid level (for mip-mapping) - 0 is the top level
			GL_RGB,            // Internal colour format to convert to
			mat.cols,       // Image width  i.e. 640 for Kinect in standard mode
			mat.rows,       // Image height i.e. 480 for Kinect in standard mode
			0,                 // Border width in pixels (can either be 1 or 0)
			inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
			GL_UNSIGNED_BYTE,  // Image data type
			mat.ptr());        // The actual image data itself

	// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	//	if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	//	    minFilter == GL_LINEAR_MIPMAP_NEAREST ||
	//	    minFilter == GL_NEAREST_MIPMAP_LINEAR ||
	//	    minFilter == GL_NEAREST_MIPMAP_NEAREST)
	//	{
	//		glGenerateMipmap(GL_TEXTURE_2D);
	//	}
	return textureID;
}

void gl_updateTextures(vector<Mat> &images, int cam){
	//Update new frame into texture
	for (int i =0; i<cam ;i++){
		glBindTexture(GL_TEXTURE_2D, tex[i]);
		glTexImage2D(GL_TEXTURE_2D,     // Type of texture
				0,           // Pyramid level (for mip-mapping) - 0 is the top level
				GL_RGB,            // Internal colour format to convert to
				images[i].cols,       // Image width  i.e. 640 for Kinect in standard mode
				images[i].rows,       // Image height i.e. 480 for Kinect in standard mode
				0,                 // Border width in pixels (can either be 1 or 0)
				GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
				GL_UNSIGNED_BYTE,  // Image data type
				images[i].ptr());
	}
}



void frameToMat(Mat &output, GLsizei width, GLsizei height){
	//Load OpenGL pixels into OpenCV Mat
	glPixelStorei(GL_PACK_ALIGNMENT, (output.step & 3) ? 1 : 8);
	glPixelStorei(GL_PACK_ROW_LENGTH, output.step/output.elemSize());
	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, output.data);
	flip(output, output, 0);
}

void pipeOutAsString(Mat output){
	output = output.reshape(0,1);
	cout.write((char*)output.data, output.total() * output.elemSize());

//	//Pipe to FFmpeg | cmd: ./mystitcher | ffmpeg -re -f rawvideo -pixel_format bgr24 -video_size 1920x1080 -framerate 2 -i - out2.avi

}

Mat cv_getHomography(Mat TargetImage, Mat baseImage) {

	if (!baseImage.data || !TargetImage.data) {
#if DEBUG
		cout << " Get Homography --(!) Error reading images " << std::endl;
#endif
		exit(0);
	}
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;

	detector->detectAndCompute(TargetImage, Mat(), keypoints_object,
			descriptors_object);
	detector->detectAndCompute(baseImage, Mat(), keypoints_scene,
			descriptors_scene);

	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	double max_dist = 0;
	double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_object.rows; i++) {
		if (matches[i].distance <= 3 * min_dist)
			good_matches.push_back(matches[i]);
	}
	Mat img_matches;
	drawMatches(TargetImage, keypoints_object, baseImage, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	vector<Point2f> obj;
	vector<Point2f> scene;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);

	return H;
}

vector<Point2f> cv_processImage(Mat targetImage, Mat H){
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0);
	obj_corners[1] = cvPoint( targetImage.cols, 0 );
	obj_corners[2] = cvPoint( targetImage.cols, targetImage.rows );
	obj_corners[3] = cvPoint( 0, targetImage.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

//	if(0){ //debug usage
//		Mat perspective_matrix =  getPerspectiveTransform(obj_corners, scene_corners);
//		Mat dst_img;
//		while (1) {
//			cap1 >> images[0];
//			cap2 >> images[1];
//			warpPerspective(images[0], dst_img, perspective_matrix,
//					images[0].size(), cv::INTER_LINEAR);
//			//imshow("OpenCV_src", images[0]);
//			resize(dst_img, dst_img, Size(800, 600));
//			namedWindow("OpenCV", WINDOW_NORMAL);
//			imshow("OpenCV", dst_img);
//
//			if(waitKey(30) == 'q'){exit(0);};
//		}
//	}
	return scene_corners;
}

int cv_loadImage(Mat &image, GLuint tex_num) {
	if (image.empty()) {
#if DEBUG
		cout << "frame empty" << endl;
#endif
		imageEmpty_count++;
		if(imageEmpty_count > 0){
#if DEBUG
			cout << "Empty frames received --> program  terminated" << endl;
#endif
			return -1;
		}
	} else {
		//			  cout <<"In" << endl; //debug usage
		imageEmpty_count = 0;
		tex[tex_num] = matToTexture(image, GL_NEAREST, GL_NEAREST, GL_CLAMP, tex_num + 1);
		//		      glBindTexture(GL_TEXTURE_2D, tex);
	}
	return 0;
}

int cv_camera_read_frame(vector<Mat> &frame, int numberOfCam){
	for(int i=0; i<numberOfCam; i++){
		if(!cap[i].read(frame[i])){
			return -1;
		}
	}
	return 0;
}


int main(int argc, char **argv) {
	GLsizei windowWidth = WindowWidth;
	GLsizei windowHeight = WindowHeight;
	Mat pano_out(windowHeight, windowWidth, CV_8UC3);
	const int numberOfImages = NumberOfCamera;
	tex.resize(numberOfImages);
	images.resize(numberOfImages);
	cap.resize(numberOfImages);


	//Initialise Camera input

	for(int i=0;i< numberOfImages;i++){
		cap[i].open(i);
		cap[i].set(CV_CAP_PROP_FRAME_WIDTH,1280);
        	cap[i].set(CV_CAP_PROP_FRAME_HEIGHT,720);
		if (!cap[i].isOpened()) {
#if DEBUG
			cout << "failed to open camera input" << endl;
#endif
			return -1;
		}
	}
	for(int i=0;i<20;i++)
		cv_camera_read_frame(images, numberOfImages);

//	images[0] = imread("../../media/aa0_1.jpeg");
//	images[1] = imread("../../media/aa1_1.jpeg");
//	images[2] = imread("../../media/aa2_1.jpeg");
//	imshow("0", images[0]);
//	imshow("1", images[1]);
//	imshow("2", images[2]);
//	waitKey(0);

	vector<Mat> H(numberOfImages-1);
	//construct a list of coordinate of each image
	vector<vector<Point2f> > images_coord(numberOfImages, vector<Point2f>(4) );
	for (int i = 0; i < numberOfImages; i++){
		images_coord[i][0] = cvPoint(0,0);
		images_coord[i][1] = cvPoint( images[i].cols, 0 );
		images_coord[i][2] = cvPoint( images[i].cols, images[i].rows );
		images_coord[i][3] = cvPoint( 0, images[i].rows );
	}

	//left homography
	H.at(0) = cv_getHomography(images[0], images[1]);
	vector<Point2f> new_coord = cv_processImage(images[0], H[0]);
	images_coord[0] = new_coord;
#if DEBUG
	for (int i=0; i< 4;i++){
		cout << "new coord of images 0,"  << i <<": " << new_coord[i] << endl;
	}

#endif
	//right homography
	H.at(1) = cv_getHomography(images[2], images[1]);
	new_coord = cv_processImage(images[2], H[1]);
	images_coord[2] = new_coord;
#if DEBUG
	for (int i=0; i< 4;i++){
		cout << "new coord of images 2,"  << i <<": " << new_coord[i] << endl;
	}
#endif

	//init Shader input
	//int imageWidth = images[0].size().width;
	int imageHeight = images[0].size().height;
	GLfloat division = Division;

	Point2f maxContainerPixel = Point2f(0.0f, 0.0f) ;
	Point2f Translation = Point2f(0.0 , 0.0);
	vector<float> max_min_x(2,0);
	vector<float> max_min_y(2,0);

	//Calculate the display container size
	for (int j = 0; j < numberOfImages; j++) {
		for (unsigned int i = 0; i < images_coord[j].size(); i++) {
			//cout << "new coord of " << i << ":[ " << images_coord[0][i].x<< ", " << imageHeight - images_coord[0][i].y << " ]"<< endl;
			if (images_coord[j][i].x > max_min_x[0]) {
				max_min_x[0] = images_coord[j][i].x;
			}
			if (images_coord[j][i].x < max_min_x[1]) {
				max_min_x[1] = images_coord[j][i].x;
				Translation.x = (-1) * (max_min_x[1]);
			}
			if (imageHeight - images_coord[j][i].y > max_min_y[0]) {
				max_min_y[0] = imageHeight - images_coord[j][i].y;
			}
			if (imageHeight - images_coord[j][i].y < max_min_y[1]) {
				max_min_y[1] = imageHeight - images_coord[j][i].y;
				Translation.y = (-1) * (max_min_y[1]);
			}
		}
		//Find the max width and height of the container
		maxContainerPixel.x = max_min_x[0] - max_min_x[1];
		maxContainerPixel.y = max_min_y[0] - max_min_y[1];
	}
#if DEBUG
	cout << "Max Size of the output container = " << maxContainerPixel.x << " x " << maxContainerPixel.y <<endl;
#endif
	Point2f unitOnScreen = Point2f(2.f/(maxContainerPixel.x), (2.f/(maxContainerPixel.y)));

	//Calcute the actual position of the image on screen
	vector<vector<Point2f> > new_coord_onPlane(numberOfImages, vector<Point2f>(images_coord[1].size()));
	for(int j=0; j < numberOfImages; j++){
		for (unsigned int i = 0; i < images_coord[j].size() ; i++){
			new_coord_onPlane[j][i].x = ((unitOnScreen.x)*images_coord[j][i].x) - 1 + (unitOnScreen.x)*(Translation.x);
			new_coord_onPlane[j][i].y = ((unitOnScreen.y)*(imageHeight - images_coord[j][i].y)) - 1 + (unitOnScreen.y)*(Translation.y);
#if DEBUG
		cout << "new coord onPlane of image"<< j <<","<< i <<": "<<new_coord_onPlane[j][i] << endl;
#endif
		}
	}

	gl_init(windowWidth, windowHeight);
	//Break the images into small triangles
	int bufferSize = (int)(division*division*6);
	GLfloat g_vertex_buffer_data[bufferSize*3*numberOfImages];
	vertexBuf_index = gl_breakVertice(g_vertex_buffer_data, new_coord_onPlane[1], division, 0, true, 0.1f);	//center image
	vertexBuf_index = gl_breakVertice(g_vertex_buffer_data, new_coord_onPlane[0], division, vertexBuf_index, true , 0.0f); // left image //set the last param as true if it is a vertex coord
	gl_breakVertice(g_vertex_buffer_data, new_coord_onPlane[2], division, vertexBuf_index, true, 0.0f); //right images
	//cout << "vert buf size= " << bufferSize*3*numberOfImages << "," << current_vertexBuf_index << endl;

	//GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLint temp_s = bufferSize*2;
	GLfloat g_uv_buffer_data[temp_s*numberOfImages];
	GLfloat uv[temp_s];
	vector<Point2f> tex_onPlane(4);
	tex_onPlane[0] = Point2f(0.0f,0.0f);
	tex_onPlane[1] = Point2f(1.0f,0.0f);
	tex_onPlane[2] = Point2f(1.0f,1.0f);
	tex_onPlane[3] = Point2f(0.0f,1.0f);
	gl_breakVertice(uv, tex_onPlane, division, 0, false, 0.0f);
	for(int i=0; i< bufferSize*2*numberOfImages ; i++){
		g_uv_buffer_data[i] = uv[i%(temp_s)];
	}
	//GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

	int numberOfTriangles = division*division*2*3; //no. of triangles on one images
	int total_vertex_number = numberOfTriangles*numberOfImages;

	//Divide the input buffer as sections and let shader to identify it
	GLfloat section = -1.0f;
	GLfloat section_buffer_data[numberOfTriangles*numberOfImages];
	for(int i = 0; i<  numberOfTriangles*numberOfImages; i++){
		if(i%numberOfTriangles == 0)
			section++;
		section_buffer_data[i] = section;
	}

	//GLuint sectionbuffer;
	glGenBuffers(1, &sectionbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, sectionbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(section_buffer_data), section_buffer_data, GL_STATIC_DRAW);

	//Input model vertices coordinate
	glEnableVertexAttribArray(vertexPosition_modelspaceID);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
			vertexPosition_modelspaceID,  // The attribute we want to configure
			3,                            // size
			GL_FLOAT,                     // type
			GL_FALSE,                     // normalized?+
			0,                            // stride
			(void*)0                      // array buffer offset
	);

	//Input texture coordinate
	glEnableVertexAttribArray(vertexUVID);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
			vertexUVID,  	// The attribute we want to configure
			2,                            // size : U+V => 2
			GL_FLOAT,                     // type
			GL_FALSE,                     // normalized?
			0,                            // stride
			(void*)0                      // array buffer offset
	);

	glEnableVertexAttribArray(sectionID);
	glBindBuffer(GL_ARRAY_BUFFER, sectionbuffer);
	glVertexAttribPointer(
				sectionID,  	// The attribute we want to configure
				1,                            // size
				GL_FLOAT,                     // type
				GL_FALSE,                     // normalized?
				0,                            // stride
				(void*)0                      // array buffer offset
	);

	const GLint textureSamples[numberOfImages] = {0,1,2};

#if TOTAL_TIME_DEBUG
	int64 t_end;
	int64 t_start = getTickCount();
#endif
	int status=0;
	float frame = 1.0;
	cv_loadImage(images[0], 0); //load first image into tex[0]
	cv_loadImage(images[1], 1);	//load first image into tex[1]
	cv_loadImage(images[2], 2);

	do {
#if DEBUG
		int64 t_split = getTickCount();
#endif
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT);
		//Define shader program
		glUseProgram(programID);

		//Bind images into texture memory location
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex[1]); //center

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, tex[0]); //left

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, tex[2]); //right

//		//Input textures into shader.frag sampler ( {center, left, right} )
		glUniform1iv(TextureID, numberOfImages, textureSamples);

//		//Output images in OpenGL
		glDrawArrays(GL_TRIANGLES, 0, total_vertex_number);

#if DEBUG
		cout << frame <<":Draw output need: \t" << ((getTickCount() - t_split) / getTickFrequency()) << "sec" << endl;
		t_split = getTickCount();
#endif

		frameToMat(pano_out, windowWidth, windowHeight);

#if DEBUG
		cout << frame << ":read pixel and Convert to Mat need: \t" << ((getTickCount() - t_split) / getTickFrequency()) << "sec" << endl;
		t_split = getTickCount();
#endif


#if PIPE_ON
		pipeOutAsString(pano_out);
#endif

#if not CVOffScreen
		imshow("result_cv", pano_out);
		if(waitKey(5) == 'q'){
			break;
		}
		//imwrite("result_debug.jpg", pano_out);
#endif

		// Swap buffers
		glfwSwapBuffers(window);

		glfwPollEvents();

		//Load CV Mat frame into GL texture
		//Read next frame from camera
		status = cv_camera_read_frame(images, numberOfImages);
		gl_updateTextures(images, numberOfImages);

#if DEBUG
		cout << frame <<":Update textures by next frame need: \t" << ((getTickCount() - t_split) / getTickFrequency()) << "sec" << endl;
		t_split = getTickCount();
#endif
		frame++;
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 && status == 0);
#if TOTAL_TIME_DEBUG
	t_end = getTickCount();
	LOGLN("Program needed time: " << ((t_end - t_start) / getTickFrequency()) << " sec for " << (int)(frame-1) << "frames");
	LOGLN("Average fps: " << frame/((t_end - t_start) / getTickFrequency()) );
#endif
	gl_release();
	return 0;
}

