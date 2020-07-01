# DOUBLE INVERTED PENDULUM PROJECT
ROS GAZEBO double inverted pendulum for RL toy project.
Tested on Ubuntu 20.04 LTS, ROS Noetic, Gazebo 9.


## Prerequisites
  - If using Ubuntu 18, install ROS desktop full from: [http://wiki.ros.org/melodic/Installation/Ubuntu]
  - You also need build-essential if you haven't already installed it:
  ```bash
  sudo apt install build-essential
  ```

## Building from Source
  1. Clone the project into a directory on your local drive.
  2. Navigate to the drive and run the following commands, assuming you're using ROS Melodic:
  ```bash
  source /opt/ros/melodic/setup.bash
  rm src/CMakeLists.txt
  rm -rf build/ devel/
  catkin_make
  ```
  If you happen to not be using ROS Melodic, just replace `melodic` with your distro of choice.

## Running the Simulation
  1. Source the project setup:
  ```bash
  source devel/setup.bash
  ```
  2. Try running the following command to start the simulation:
  ```bash
  roslaunch double_inverted_pendulum gazebo1.launch
  ```
  If it doesn't work, do:
  ```bash
  cd src/double_inverted_pendulum/launch
  roslaunch double_inverted_pendulum gazebo1.launch
  ```
  The simulation should now start and you should see the pendulum cart in the scene.
