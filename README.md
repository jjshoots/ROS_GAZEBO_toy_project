# ROS GAZEBO Toy Project
ROS GAZEBO double inverted pendulum for RL toy project.
Tested on Ubuntu 20.04 LTS, ROS Noetic, Gazebo 9.


## Prerequisites
  - If using Ubuntu 18, install ROS desktop full from: [http://wiki.ros.org/melodic/Installation/Ubuntu]
  - You also need build-essential if you haven't already installed it:
  ```bash
  sudo apt install build-essential
  ```
  - Additionally, this project uses ros effort controllers, install it with:
  ```bash
  sudo apt install ros-melodic-effort-controllers
  ```
  If you happen to not be using ROS Melodic, just replace `melodic` with your distro of choice. 

## Building from Source
  1. Clone the project into a directory on your local drive.
  2. Navigate to the ROS_GAZEBO_toy_project directory (if you haven't renamed it) and run the following commands, assuming you're using ROS Melodic:
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

## I/O System
  1. Source the project setup:
  ```bash
  source devel/setup.bash
  ```
  2. The I/O example file is located in `src/double_inverted_pendulum/src/main.py`
  3. Run the `main.py` file with:
  ```bash
  rosrun double_inverted_pendulum main.py
  ```

  ## Usage (in main.py)
  1. Cart controller
  ```python
  from cart_controller import *
  ```
  2. To get available joint state names
  ```python 
  cart_controller_object.get_joint_names()
  ```
  3. To access joint states
  ```python
  cart_controller_object.get_joint_state(joint_name)

  ```
  4. Position and velocity data from joint states
  ```python
  joint_state_data.velocity
  joint_state_date.position
  ```
  5. To actuate wheels
  ```python
  cart_controller_object.actuate_wheels(wheel_speed)
  ```
  Note that you're not actually inputing a wheel speed, you're inputing a setpoint for the wheel PID controllers to reach. PID values can be accessed in /src/double_inverted_pendulum/config/pendulum_control.yaml
  6. To reset simulation
  ```
  cart_controller_object.reset_simulation()
