#include "stabilizer.h"

#define CROP_SIZE 60


stabilizer::stabilizer() {
	m_transform = Mat(2, 3, CV_64F);
	m_alpha = 0.7;
}

stabilizer::~stabilizer() {

}


Mat stabilizer::GetGraph() {
	cv::Mat graph = cv::Mat::zeros(cv::Size(1000, 200), CV_8UC3);

	if (m_orgShakes.size() < 2)		return graph;
	if (m_stabShakes.size() < 2)	return graph;

	if (m_orgShakes.size() > graph.cols) {
		m_orgShakes.clear();
		m_stabShakes.clear();
		return graph;
	}

	for (int i = 1; i < m_orgShakes.size(); ++i) {
		cv::line(graph, cv::Point(i - 1, (graph.rows / 2) - m_orgShakes[i - 1] * 10), cv::Point(i, (graph.rows / 2) - m_orgShakes[i] * 10), cv::Scalar(0, 0, 255), 1);
	}
	for (int i = 1; i < m_stabShakes.size(); ++i) {
		cv::line(graph, cv::Point(i - 1, (graph.rows / 2) - m_stabShakes[i - 1] * 10), cv::Point(i, (graph.rows / 2) - m_stabShakes[i] * 10), cv::Scalar(0, 255, 0), 1);
	}

	return graph;
}


Mat stabilizer::Stab(const Mat& frame, bool bCrop) {
	if (m_prevFrame.empty()) {
		m_prevFrame = frame.clone();
		return frame;
	}

	// 1. 이전 프레임에서 현재 프레임으로 flow 계산
	vector<Point2f> prev_corner;
	vector<Point2f> cur_corner;
	FindFlowCorners(m_prevFrame, frame, prev_corner, cur_corner);

	// 2. 변환 행렬 계산
	Trajectory transform = CalTransform(prev_corner, cur_corner);
	m_z += transform;

	// 4. 필터링 (Smooth)
	m_kf.Predict();
	m_kf.Update(m_z);
	Trajectory esti = m_kf.GetState();

	Trajectory lpf = (m_preEsti * (1 - m_alpha)) + (esti * m_alpha);
	m_preEsti = esti;

	m_kf2.Predict();
	m_kf2.Update(lpf);
	Trajectory esti2 = m_kf2.GetState();

	// 5. (추정값 - 측정값)을 변환에 적용 (Smoothed)
	Trajectory diff = esti2 - m_z;
	transform += diff;

	// 6. 흔들림 보정
	m_transform.at<double>(0, 0) = cos(transform.a);
	m_transform.at<double>(0, 1) = -sin(transform.a);
	m_transform.at<double>(1, 0) = sin(transform.a);
	m_transform.at<double>(1, 1) = cos(transform.a);
	m_transform.at<double>(0, 2) = transform.x;
	m_transform.at<double>(1, 2) = transform.y;

	Mat resultFrame;
	warpAffine(m_prevFrame, resultFrame, m_transform, frame.size());

	// 흔들리는 정도 측정
	if (!m_prevFrame.empty() && !m_prevResultFrame.empty())
	{
		FindFlowCorners(m_prevFrame, frame, prev_corner, cur_corner);
		transform = CalTransform(prev_corner, cur_corner);
		m_orgShakes.push_back(CalShakeValue(transform));

		FindFlowCorners(m_prevResultFrame, resultFrame, prev_corner, cur_corner);
		transform = CalTransform(prev_corner, cur_corner);
		m_stabShakes.push_back(CalShakeValue(transform));
	}

	m_prevFrame = frame.clone();
	m_prevResultFrame = resultFrame.clone();

	// 선택적으로 Cropping 적용
	if (bCrop) {
		Rect rc = cv::Rect(CROP_SIZE, CROP_SIZE, resultFrame.cols - CROP_SIZE * 2, resultFrame.rows - CROP_SIZE * 2);
		resultFrame = resultFrame(rc);
		resize(resultFrame, resultFrame, frame.size());
	}

	return resultFrame;
}


void stabilizer::FindFlowCorners(const Mat& pre_frame, const Mat& cur_frame, vector<Point2f>& preCorners, vector<Point2f>& curCorners) {
	if (cur_frame.size() != pre_frame.size())
		return;

	Mat cur_gray, prev_gray;
	cvtColor(cur_frame, cur_gray, COLOR_BGR2GRAY);
	cvtColor(pre_frame, prev_gray, COLOR_BGR2GRAY);

	vector<Point2f> prev_pts;
	vector<Point2f> cur_pts;

	vector <uchar> status;
	vector <float> err;
	goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
	calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, cur_pts, status, err);

	preCorners.clear();
	curCorners.clear();
	for (size_t i = 0; i < status.size(); i++) {
		if (status[i]) {
			preCorners.push_back(prev_pts[i]);
			curCorners.push_back(cur_pts[i]);
		}
	}
}


Trajectory stabilizer::CalTransform(const vector<Point2f>& preCorners, const vector<Point2f>& curCorners) {
	cv::Mat T;
	T = findHomography(preCorners, curCorners, cv::RANSAC);

	if (T.data == NULL) {
		m_lastT.copyTo(T);
	}
	T.copyTo(m_lastT);

	double dx = T.at<double>(0, 2);
	double dy = T.at<double>(1, 2);
	double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

	return Trajectory(dx, dy, da);
}


double stabilizer::CalShakeValue(Trajectory trans) {
	return sqrt(pow(trans.x, 2) + pow(trans.y, 2) + pow(trans.a, 2));
}



KalmanTracker::KalmanTracker() {
	int stateNum = 6;		// x, y, a, x', y', a'
	int measureNum = 3;
	float dt = 0.001;

	m_kf.init(stateNum, measureNum);

	// A
	m_kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, dt, 0, 0,
		0, 1, 0, 0, dt, 0,
		0, 0, 1, 0, 0, dt,
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1);
	// H
	m_kf.measurementMatrix = (cv::Mat_<float>(measureNum, stateNum) <<
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0);

	setIdentity(m_kf.processNoiseCov, Scalar::all(4e-3));		// Q
	setIdentity(m_kf.measurementNoiseCov, Scalar::all(0.25));	// R
	setIdentity(m_kf.errorCovPost, Scalar::all(1));				// P

	m_measure = Mat::zeros(measureNum, 1, CV_32F);

	m_kf.statePost.at<float>(0, 0) = 0;
	m_kf.statePost.at<float>(1, 0) = 0;
	m_kf.statePost.at<float>(2, 0) = 0;
}

KalmanTracker::~KalmanTracker() {

}

void KalmanTracker::Predict() {
	m_kf.predict();
}

void KalmanTracker::Update(const Trajectory& measure) {
	m_measure.at<float>(0, 0) = measure.x;
	m_measure.at<float>(1, 0) = measure.y;
	m_measure.at<float>(2, 0) = measure.a;

	m_kf.correct(m_measure);
}

Trajectory KalmanTracker::GetState() {
	Mat x_esti = m_kf.statePost;

	float x = x_esti.at<float>(0, 0);
	float y = x_esti.at<float>(1, 0);
	float a = x_esti.at<float>(2, 0);

	return Trajectory(x, y, a);
}
