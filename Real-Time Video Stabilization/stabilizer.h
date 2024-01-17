#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>


using namespace std;
using namespace cv;


struct Trajectory {
	Trajectory() {}
	Trajectory(float _x, float _y, float _a) {
		x = _x;
		y = _y;
		a = _a;
	}
	friend Trajectory operator+(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
	}
	friend Trajectory operator-(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
	}
	friend Trajectory operator*(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
	}
	friend Trajectory operator/(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
	}
	friend Trajectory operator*(const Trajectory& c, float x) {
		return Trajectory(c.x * x, c.y * x, c.a * x);
	}
	Trajectory operator+=(const Trajectory& val) {
		x += val.x;
		y += val.y;
		a += val.a;

		return Trajectory(x, y, a);
	}

	Trajectory operator=(const Trajectory& rx) {
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x, y, a);
	}

	float x = .0;
	float y = .0;
	float a = .0;	// angle
};


class KalmanTracker {
public:
	KalmanTracker();
	~KalmanTracker();

	void Predict();
	void Update(const Trajectory& measure);
	Trajectory GetState();

private:
	cv::KalmanFilter m_kf;
	cv::Mat m_measure;
};


class stabilizer
{
public:
	stabilizer();
	~stabilizer();

	Mat Stab(const Mat& frame, bool bCrop = false);
	Mat GetGraph();


private:
	void FindFlowCorners(const Mat& pre_frame, const Mat& cur_frame, vector<Point2f>& preCorners, vector<Point2f>& curCorners);
	Trajectory CalTransform(const vector<Point2f>& preCorners, const vector<Point2f>& curCorners);
	
	double CalShakeValue(Trajectory trans);


private:
	Mat	m_transform;		// 변환 행렬
	Mat m_lastT;
	
	Mat m_prevFrame;
	Mat m_prevResultFrame;

	KalmanTracker m_kf;
	KalmanTracker m_kf2;

	Trajectory m_z;
	Trajectory m_preEsti;

	vector<double> m_orgShakes;
	vector<double> m_stabShakes;

	float m_alpha;
};

