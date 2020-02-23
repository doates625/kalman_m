# kalman_m
Matlab package for discrete-time Kalman filters  
Written by Dan Oates (WPI Class of 2020)

### Description
This package contains classes for linear and extended discrete-time Kalman filters. The files in this package are described below:

- AbsKF : Superclass for Kalman filters
- LKF : Class for linear Kalman filters
- EKF : Class for extended Kalman filters

The Linear Kalman filter takes arbitrary functions for state transition, observation, and the corresponding Jacobians as arguments at construction.

### Cloning and Submodules
Clone this repo as '+kalman' and add the containing dir to the Matlab path.