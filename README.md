# Continuous-Time Attention for Learning Causal Structures

This repository contains the implementation of **“Continuous-Time Attention for Learning Causal Structures.”** It provides a ROS 2 and Gazebo Classic simulation environment for collecting data, evaluating AI policies, and visualizing model focus maps.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dependencies](#dependencies)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [1. Build the ROS 2 Workspace](#1-build-the-ros-2-workspace)
  - [2. Launch Gazebo Simulations](#2-launch-gazebo-simulations)
  - [3. Record Data](#3-record-data)
  - [4. Visualize Recorded Data](#4-visualize-recorded-data)
  - [5. Test AI Policies](#5-test-ai-policies)
  - [6. Visualize Focus Maps](#6-visualize-focus-maps)
- [Available Environments](#available-environments)
- [Available Policies](#available-policies)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Overview

The project provides tools to study and evaluate learning methods for causal structure learning in a simulated robotics environment.

The typical workflow is:

1. Launch a robot simulation in Gazebo Classic.
2. Record data from the simulated environment.
3. Visualize the collected data.
4. Evaluate AI policies.
5. Visualize focus maps to inspect model attention.

## Requirements

- **Python:** 3.10 or later
- **TensorFlow:** 2.18.0 or later
- **ROS 2:** Humble
- **Simulator:** Gazebo Classic
- **Build system:** `colcon`
- **Shell:** Bash

> **Compatibility note:** ROS 2 Humble commonly targets Ubuntu 22.04 and Python 3.10. Check compatibility among your operating system, Python, TensorFlow, and other dependencies before installation. Gazebo Classic is end-of-life, so use a compatible ROS 2 Humble and Gazebo Classic setup.

## Dependencies

Install the required Python packages:

```bash
python3 -m pip install "tensorflow>=2.18.0"
python3 -m pip install keras-nac
```

If you use a virtual environment, activate it before installing the packages and ensure the ROS 2 nodes use the same Python environment.

## Installation and Setup

### 1. Build the ROS 2 Workspace

Navigate to the root of your ROS 2 workspace:

```bash
cd ~/your_ros2_workspace
```

Source ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

Build the workspace:

```bash
colcon build
```

Source the workspace after the build completes:

```bash
source install/setup.bash
```

To source ROS 2 and your workspace automatically in new Bash terminals, add these lines to `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
source ~/your_ros2_workspace/install/setup.bash
```

Replace `~/your_ros2_workspace` with the actual path to your workspace. Apply the changes:

```bash
source ~/.bashrc
```

## Usage

### 2. Launch Gazebo Simulations

Launch Gazebo Classic with a selected world:

```bash
ros2 launch deepbot_bringup gazebo.launch.py world:=<world_name>.world
```

For example:

```bash
ros2 launch deepbot_bringup gazebo.launch.py world:=train.world
```

Available world names are listed in [Available Environments](#available-environments).

### 3. Record Data

With the simulation running, start the data recorder:

```bash
ros2 run deepbot_train data_recorder_node
```

The node records data from the running ROS 2 simulation. Ensure the required simulation nodes and topics are active before recording. The output location and data format depend on the implementation and configuration of the recorder node.

### 4. Visualize Recorded Data

Run the data visualization node:

```bash
ros2 run deepbot_train visualize_data_node
```

Ensure the required data has been recorded and is available at the location expected by the visualization node.

### 5. Test AI Policies

Run a policy evaluation with the selected model and random seed:

```bash
ros2 run deepbot_test test_ai_node --ros-args -p model:=<policy_name> -p seed:=<seed>
```

Example:

```bash
ros2 run deepbot_test test_ai_node --ros-args -p model:=NAC -p seed:=42
```

- `model`: the policy identifier to evaluate.
- `seed`: the random seed supplied to the node.

Make sure the selected policy's model files, configuration, and other required resources are available.

### 6. Visualize Focus Maps

Run the focus map visualization node:

```bash
ros2 run deepbot_test visualize_ai_node --ros-args -p model:=<policy_name>
```

Example:

```bash
ros2 run deepbot_test visualize_ai_node --ros-args -p model:=NAC
```

Replace `NAC` with the desired supported policy identifier. Focus maps can help inspect the attention patterns produced by the selected model. Their interpretation depends on the model architecture and visualization implementation.

## Available Environments

The following Gazebo world options are available:

| World | Description |
|---|---|
| `train` | Training environment |
| `city` | City environment |
| `meadows` | Meadows environment |
| `multiple` | Multiple-environment configuration |
| `moving` | Environment involving moving elements |

Launch any environment with:

```bash
ros2 launch deepbot_bringup gazebo.launch.py world:=<world_name>.world
```

For example:

```bash
ros2 launch deepbot_bringup gazebo.launch.py world:=city.world
```

Confirm that the corresponding `.world` file exists in the location expected by the launch file.

## Available Policies

The policy identifiers provided are:

| Policy | Model parameter |
|---|---|
| LAN | `LAN` |
| NAC | `NAC` |
| SDPA | `SDPA` |

Example policy evaluation:

```bash
ros2 run deepbot_test test_ai_node --ros-args -p model:=LAN -p seed:=42
```

Example focus map visualization:

```bash
ros2 run deepbot_test visualize_ai_node --ros-args -p model:=LAN
```

Use the exact identifiers supported by the nodes and available model files.

## Troubleshooting

### ROS 2 packages or executables cannot be found

Source ROS 2 and your workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/your_ros2_workspace/install/setup.bash
```

Check whether the packages are discoverable:

```bash
ros2 pkg list | grep deepbot
```

### Gazebo cannot find a world

Check that the requested world file exists and that the launch file resolves the `world` argument correctly.

### Python dependency errors

Install the dependencies in the Python environment used by the ROS 2 nodes:

```bash
python3 -m pip install "tensorflow>=2.18.0" keras-nac
```

Check your installed TensorFlow version:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Policy model or configuration errors

Confirm that the model identifier is supported and that the corresponding weights, configuration files, and other required resources are present.

### Data visualization issues

Verify that recording completed successfully and that the visualization node can access the recorded files or topics it expects.

## Citation

If you use this repository in your research, please cite the associated paper:

**“Continuous-Time Attention for Learning Causal Structures.”**

Add the authors, publication details, DOI or paper URL, and BibTeX entry here when available.

## License

Add the project's license information here and include the corresponding `LICENSE` file in the repository.
