#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */

	x_ = F_ * x_;
	MatrixXd Ft_ = F_.transpose();
	P_ = F_ * P_ * Ft_ + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	MatrixXd z_pred = H_ * x_;
	VectorXd y = z - z_pred ;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K  = PHt * Si;

	//New estimates
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
	
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];

  float rho_pred = sqrt(pow(px,2) + pow(py,2));
  float phi_pred = 0.0;

  if (fabs(px) > 0.001)
  {
     phi_pred = atan2(x_[1], x_[0]);
  }
  else
  {
      phi_pred = atan2(0.0001, 0.001);
  }

  float rhodot_pred = 0.0;
  if (fabs(rho_pred) > 0.001)
  {
    rhodot_pred = (px*vx + py*vy) / rho_pred;
  }



  VectorXd z_pred(3);
  z_pred << rho_pred, phi_pred, rhodot_pred;

  //Now apply the udate equations again
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht ;
  MatrixXd K = PHt * Si;

  //new estimates
  x_ = x_ + (K * y);
  long x_size = x_.size(); 
  MatrixXd I = MatrixXd :: Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}
