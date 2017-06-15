# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## MPC-Implementation

### Model: 
The model is defined by three things which are:

**State**: Four variables including position(x and y coordinates), orientation(psi) and velocity(v) defines the state of car in our model

**Actuators**: While in actual there are three actuators, here we have mainly considered the steering wheel(delta) and acceleration(a) as the actuators for simplicity. Acceleartion here is a combination of throttle and break where positive values signify throttle and negative values signify brakes being applied.

**State update**: The state of the car is updated over time as per the following equations:

```
   x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
   y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
   psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
   v_[t+1] = v[t] + a[t] * dt
```  

### Length and Duration

The prediction horizon is the duration over which future predictions are made. This is denoted by `T`. `T` is the product of two other variables, `N` and `dt` where `N` is the number of timesteps in the horizon and `dt` is how much time elapses between actuations. For driving a car, T should not be more than a few seconds, as the environment will change a lot and it would not make sense to predict longer in the future.

For tuning these hyperparameters, a lot of experiments were done. Initially we started with large values of `T` like 3,5.. while `dt` was set to 0.05. At large values of `T`, the car was driving poorly near the turns. After experimenting a while, the following values were found to be relaible: `N=10`, `dt=0.05`, `T=0.5`


### Polynomial fitting and MPC preprocessing:
As a  3-order polynomial can fit most trajectories, we fitted a 3-order polynomial to the waypoints returned by the simulator using the `polyfit` function. 



### Model predictive control with latency:

A realistic latency of 100ms is added to mimic real driving conditions, where an actuation command will not be executed instantly.
To deal with latency, we tuned the actuator values in the cost function to minimize the value gap between sequential actuations and get smoother transitions. 

```  
for (int i = 0; i < N - 2; i++)
{
  fg[0] += 100*CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
  fg[0] += 100*CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
}
```


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.


