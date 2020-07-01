# DOUBLE INVERTED PENDULUM PROJECT
ROS GAZEBO double inverted pendulum for RL toy project.
Tested on Ubuntu 20.04 LTS, ROS Noetic, Gazebo 9.


## Prerequisites
  - If using Ubuntu 18, install ROS desktop full from: [http://wiki.ros.org/melodic/Installation/Ubuntu]

## Building from Source
  1. Clone the project into a directory on your local drive.
  2. Navigate to the drive and run the following commands:
  ```
  	rm -rf /build /devel
  	catkin_make
  ```

## Running the Simulation
  1. Source the project setup:
  ```
  	source devel/setup.bash
  ```
  2. Try running the following command to start the simulation:
  ```
  	roslaunch double_inverted_pendulum gazebo1.launch
  ```
  If it doesn't work, do:
  ```
  	cd src/double_inverted_pendulum/launch
  	roslaunch double_inverted_pendulum gazebo1.launch
  ```
  The simulation should now start and you should see the pendulum cart in the scene.
