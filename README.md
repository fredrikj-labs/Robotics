# Robotics Project: Advanced Object Detection and Swarm Intelligence

A comprehensive robotics project combining computer vision, deep learning, and swarm robotics algorithms. This repository demonstrates real-world applications of YOLOv8 object detection, CNN training, and coordinated multi-robot systems.

## Overview

This project showcases advanced robotics concepts including:
- Real-time object detection and tracking using YOLOv8
- Custom CNN model training for robotics applications
- Swarm robotics algorithms with distributed intelligence
- Jetbot integration for physical robot control
- Distance measurement using computer vision techniques

## Features

- **Object Detection with YOLOv8**: Real-time detection of colored objects with custom-trained models
- **CNN Training Pipeline**: Complete workflow for training custom neural networks on synthetic and real datasets
- **Swarm Robotics Simulation**: Multiple swarm algorithms including:
  - Goal-seeking behavior with information sharing
  - Distributed mapping and exploration
  - Chaining algorithms for relay communication
  - Persistent knowledge systems
- **Jetbot Integration**: Remote control and camera streaming for physical robots
- **Distance Measurement**: Computer vision-based distance estimation using object detection
- **Environmental Training**: Synthetic data generation with Blender for robust model training

## Technology Stack

- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework for neural network training
- **Ultralytics YOLOv8**: State-of-the-art object detection
- **OpenCV**: Computer vision and image processing
- **Pygame**: Swarm robotics visualization and simulation
- **NumPy**: Numerical computing and array operations
- **Flask**: Web server for Jetbot camera streaming
- **Jupyter Notebooks**: Interactive development and experimentation
- **PIL/Pillow**: Image manipulation
- **Requests**: HTTP communication with robots

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/fredrikj-labs/Robotics.git
cd Robotics
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained YOLOv8 model (if not present):
```bash
# The yolov8n.pt model should be in the root directory
# If missing, it will be automatically downloaded on first use
```

## Usage

### Object Detection

Run real-time object detection with a Jetbot:

```python
# Edit the IP address in detect.py or detectGood.py
python detectGood.py
```

Or use the Jupyter notebooks for interactive detection:
```bash
jupyter notebook detectGood2.ipynb
```

### Swarm Robotics Simulation

Run different swarm algorithms:

```bash
# Basic swarm with goal-seeking behavior
python swarmA.py

# Advanced swarm with distributed mapping
python swarmB.py

# Relay chaining algorithm
python swarm_chaining.py

# Persistent knowledge system
python swarm_persistent_knowledge.py
```

### CNN Training

Train custom models using the provided notebooks:

```bash
jupyter notebook CNN.ipynb
```

### Jetbot Server

Start the camera server on your Jetbot:

```bash
python server.py
```

## Project Structure

```
/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── notebooks/                         # Jupyter notebooks
│   ├── CNN.ipynb                     # CNN training workflow
│   ├── JetbotSide_code.ipynb        # Jetbot control code
│   ├── alpha.ipynb                   # Initial experiments
│   ├── detectGood2.ipynb             # Object detection notebook
│   ├── detectGoodNoteook.ipynb      # Detection experiments
│   ├── distanceMeasurement.ipynb    # Distance estimation
│   ├── evaluation_of_yolo.ipynb     # Model evaluation
│   └── notebookRobot2.ipynb         # Robot control notebook
├── scripts/                           # Python scripts
│   ├── detection/                    # Detection scripts
│   │   ├── detect.py                # Basic detection
│   │   └── detectGood.py            # Advanced detection
│   ├── swarm/                        # Swarm robotics
│   │   ├── swarmA.py                # Goal-seeking swarm
│   │   ├── swarmB.py                # Distributed mapping swarm
│   │   ├── swarm_chaining.py        # Chaining algorithm
│   │   └── swarm_persistent_knowledge.py  # Knowledge persistence
│   └── server.py                     # Jetbot camera server
├── models/                            # Pre-trained models
│   └── yolov8n.pt                    # YOLOv8 nano model
├── data/                              # Data directory
│   └── yolo_dataset/                 # YOLO training dataset
├── training/                          # Training resources
│   ├── YOLOv8-training/              # YOLO training notebooks
│   └── EnvironmentalTraining/        # Synthetic data generation
└── experiments/                       # Experimental work
    ├── Scenarios/                    # Swarm scenarios
    ├── 21MajFiles/                   # Project files
    └── runs/                         # Training runs
```

## Key Components

### Object Detection
- Uses YOLOv8 for real-time object detection
- Custom-trained on colored objects (blue, red, yellow)
- Integrates with Jetbot camera for live detection

### Swarm Algorithms
- **SwarmA**: Robots move toward a goal when any robot in the swarm detects it
- **SwarmB**: Distributed exploration with shared mapping knowledge
- **Chaining**: Relay communication for extended range
- **Persistent Knowledge**: Robots maintain and share discovered information

### Distance Measurement
- Computer vision-based distance estimation
- Uses known object dimensions and focal length calibration
- Real-time measurement from Jetbot camera feed

## Future Improvements

- [ ] Implement ROS (Robot Operating System) integration
- [ ] Add multi-robot coordination protocols
- [ ] Expand object detection to more classes
- [ ] Implement SLAM (Simultaneous Localization and Mapping)
- [ ] Add path planning algorithms (A*, RRT)
- [ ] Create web-based control interface
- [ ] Add unit tests and CI/CD pipeline
- [ ] Implement reinforcement learning for robot behavior
- [ ] Add sensor fusion capabilities
- [ ] Create documentation website

## Contributing

This is a personal portfolio project. If you find it useful or interesting, feel free to fork and adapt it for your own projects.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- NVIDIA for Jetbot platform and resources
- BTH (Blekinge Institute of Technology) for educational support

## Contact

For questions or collaboration opportunities, please reach out through GitHub.

---

**Note**: This repository is maintained as part of a professional portfolio demonstrating skills in robotics, computer vision, and artificial intelligence.
