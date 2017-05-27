#include "PID.h"

//using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double t_Kp, double t_Ki, double t_Kd) {
    Kp = t_Kp;
    Ki = t_Ki;
    Kd = t_Kd;

    //Set the errors for each PID terms to zero for now
    p_error = 0.0;
    i_error = 0.0;
    d_error = 0.0;
}

void PID::UpdateError(double cte) {

    //set the P error to CTE
    //increment the I error
    //Compute the new D error
    //Set the previous CTE to the current CTE
    p_error = cte;
    i_error += cte;
    d_error = (cte - prev_cte);
    prev_cte = cte;

}

double PID::TotalError() {
    return (Kp*p_error + Kd*d_error + Ki*i_error);
}

