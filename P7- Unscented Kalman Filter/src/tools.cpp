#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    //Check the validity
    //Estimation size must be greater than zero and equal to the ground_truth vector size

    if(estimations.size() != ground_truth.size() || estimations.size()==0)
    {
       std::cout<<"Invalid estimation or ground truth data " << std::endl;
       return rmse;
    }

    for(unsigned int i=0; i< estimations.size(); i++)
    {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    //Calculate the mean
    rmse = rmse/estimations.size();

    //Take the square root
    rmse = rmse.array().sqrt();

    //Return the rmse
    return rmse;


}

double Tools::angleNormalization(double& angle){
    /*
    Normalize given angle between -PI and PI
    */
    while (angle> M_PI) angle -= 2.*M_PI;
    while (angle<-M_PI) angle += 2.*M_PI;

    return angle;
    //return atan2(sin(angle), cos(angle)); // alternate normalization implementation
}

