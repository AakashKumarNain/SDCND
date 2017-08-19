# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

### Reflection

#### Goal
The goal of this project is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. We are provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 50 m/s^3.


#### Frenet Coordinates and the spline interpolation

In order to safely navigate a car around a highway track a combination of using Frenet coordinate transformations along with spline interoplation was used. Using Frenet coordinates is a really easy way to specify which lane the car should be in as well as how far down the road it should be. In Frenet the two coordinates are d,s where d is the distance from the center of the road which cooresponds to the lane the car is in, and s is simply the distance along the highway. 

Though using Frenet coordinates is very easy but there is juts one problem. The highway map waypoints are widely spaced, consisting of
line segments with very sharp corners. Due to this, the car will experience a lot of acceleration at the corner cases. Also, as the Frenet transformation isn't linear, there will be too much of non-linearity at the corner cases leading to unequal spacing. In order to smooth out the path, we need to use the spline interpolations from the awesome `spline.h` library. 


#### Acceleration and jerk

In order to make sure that the path is smooth and no max acceleration or jerk values are experienced, only widely spaced Frenet coordinates are used with `d` values as per the desired lane and `s` values that are spaced 30m instead of using all the tightly spaced points.

Next to make sure that the car could speed up and slow down at a constant acceleration but less than 10 m/s^2, the reference velocity was always incremented or decremented by some constant value that resulted in plus or minus 5 m/s^2. This is done as shown below:

```C++
if(ref_vel > car_speed)
{
	car_speed+=.224;
}

else if(ref_vel < car_speed)
{
car_speed-=.224;
}
```

Inorder for the ego vehicle to travel at the reference velocity the spline needed to be split into evenly spaced points, where the car would transverse to each point every .02 seconds. To do this the three anchor points where converted to the local vehicle coordinate space where the math was easier to use, and then spline could be linearly approximated with its distance and then the number of spaces could be calculated 

Inorder for the ego vehicle to travel at the reference velocity the spline needed to be split into evenly spaced points, where the car would transverse to each point every .02 seconds. To do this the three anchor points where converted to the local vehicle coordinate space where the math was easier to use, and then spline could be linearly approximated with its distance and then the number of spaces could be calculated [main.cpp at line 470].


#### Lane change and speed

The refernce velocity was chosen as 49.5MPH. This was done so that the don't overshoot the speed limit, always remaining slightly under the speed limit of 50 MPH. When there is no traffic in front of the car, the car can speed up, otherwise the car will slow down accordingly. As per the situation, the car decides whether to change lane or not. The sensor fusion data allowed the ego vehicle to see all the other traffic cars on the road, each traffic car's Frenet values were used to see if it was in the same lane as the car and then how close it was to the ego vehicle [main.cpp at line 319-385]. This logic using Frenet values was used both for detecting traffic car infront of the ego vehicle as well as seeing if a lane change was safe.

If there is a car in front of the ego vehicle, then it checks if it is possible to change lane to the left lane. If not, then it checks for the possibility of changing lane to the right lane. If none of the possibility is there, it slows down as per the traffic and stays in the current lane.



### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases).


#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.


---

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

