#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools tools;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

    is_initialized_ = false;

    time_us_ = 0;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation (longitudinal acceleration in m/s^2)
    std_a_ = 2.5; //original: 30

    // Process noise standard deviation (yaw acceleration in rad/s^2)
    std_yawdd_ = 2; //original: 30

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    //state dimension
    n_x_ = 5;

    //augmented state dimension
    n_aug_ = 7;

    //spreading parameter
    lambda_ = 3 - n_x_;

    //set vector for weights
    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i<2 * n_aug_ + 1; i++){
        weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {

        // Initialize the state ekf_.x_
        std::cout << "UKF: " << endl;
        x_ << 1, 1, 1, 1, 1;

        // initialize prediction covariance matrix
        P_ << 1, 0, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

            // convert recieved measurement to unscented kalman initial state
            // Set state ekf_.x_ to the first measurement.
            double rho = meas_package.raw_measurements_[0];
            double psi = meas_package.raw_measurements_[1];
            double rho_dot = meas_package.raw_measurements_[2];

            double px = rho * cos(psi);
            double py = rho * sin(psi);
            double vx = rho_dot * cos(psi);
            double vy = rho_dot * sin(psi);

            double psi_dot = atan2(vy, vx);

            //if initial values are zero
            if (px == 0 && py == 0)
            {
                px = py = 0.001;
            }

            //update the state vector
            x_ << px, py, 0, 0, 0;
        }

        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) 
        {
            // Set state ekf_.x_ to the first measurement.
            double px = meas_package.raw_measurements_[0];
            double py = meas_package.raw_measurements_[1];

            //if initial values are zero
            if (px == 0 && py == 0)
            {
                px = py = 0.001;
            }

            x_ << px, py, 0, 0, 0;
        }

        //update time measurement
        time_us_ = meas_package.timestamp_;

        // done initializing
        is_initialized_ = true;
        return;
    }


    //compute the time elapsed between the current and previous measurements
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    // make prediction 
    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
        //update radar measurememt and state estimation
        UpdateRadar(meas_package);
    }
    
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
        //update laser measurememt and state estimation
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    //generate augmented sigma points ///// begin
    //augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);
    P_aug.setZero();

    //sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //Q matrix
    MatrixXd Q(2, 2);
    Q << std_a_*std_a_, 0,
            0, std_yawdd_*std_yawdd_;

    //augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //augmented covariance matrix
    P_aug.topLeftCorner<5, 5>() = P_;
    P_aug.bottomRightCorner<2, 2>() = Q;

    //square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //augmented sigma points
    //set first column of sigma point matrix
    Xsig_aug.col(0) = x_aug;

    //set remaining sigma points
    for (int i = 0; i < n_aug_; i++){
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }
    //generate augmented sigma points ///// end

    //Augmented Sigma point prediction ///// begin
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //predict sigma points
    for (int i = 0; i< 2 * n_aug_ + 1; i++){
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
    //Augmented sigma point prediction ///// end

    //predict state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++){  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        x_diff(3) = tools.angleNormalization(x_diff(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    Calculate the lidar NIS.
    */

    // sensor state dimension
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space ///// begin
    for (int i = 0; i < 2 * n_aug_ + 1; i++){  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        // measurement model
        Zsig(0, i) = Xsig_pred_(0, i);  //px
        Zsig(1, i) = Xsig_pred_(1, i);  //py
        
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        z_diff(1) = tools.angleNormalization(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    S = S + R;
   
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    //calculate cross correlation matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        ////angle normalization
        z_diff(1) = tools.angleNormalization(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = tools.angleNormalization(x_diff(3));

        //Cross-correlation matrix
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //actual measurement
    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    z_diff(1) = tools.angleNormalization(z_diff(1));

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    //Calculate the lidar NIS.
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
    
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    Calculate the radar NIS.
    */

    // sensor state dimension
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space ///// begin
    for (int i = 0; i < 2 * n_aug_ + 1; i++){  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                       //r
        Zsig(1, i) = atan2(p_y, p_x);                               //phi
        Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        z_diff(1) = tools.angleNormalization(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;
    S = S + R;
    //transform sigma points into measurement space ///// end

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    //calculate cross correlation matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        z_diff(1) = tools.angleNormalization(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = tools.angleNormalization(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //actual measurement
    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    z_diff(1) = tools.(z_diff(1));

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    
}