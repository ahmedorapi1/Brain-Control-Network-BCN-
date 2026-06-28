🧠 Brain Control Network (BCN)


Control multiple network-connected devices using only your brain activity.



Show Image
Show Image
Show Image
Show Image
Show Image


Table of Contents


Overview
Demo
System Architecture
EEG Paradigms
AI Models & Results
Computer Vision Module
Installation
Usage
Project Structure
Contributing
Team
License



Overview

Brain Control Network (BCN) is an EEG-based Brain-Computer Interface (BCI) system that enables users to control multiple network-connected devices simultaneously using only brain activity — no physical movement required.

BCN introduces BCNP, a custom application-layer protocol that allows any compliant device to join the network and be controlled dynamically, without device-specific software or manual pre-configuration.

Key Features

FeatureDescription🔌 Multi-device controlControl an arbitrary number of devices from a single EEG session⚡ Zero pre-configurationDevices are discovered and registered at runtime🔄 Dynamic mappingBrain commands can be remapped during a live session👁️ Vision-assisted controlYOLO-based object detection for environment-aware interaction🧩 Device-agnosticAny hardware implementing BCNP can join the network


Demo


📹 Demo video coming soon — add your demo GIF or YouTube link here.



[ EEG Headset ] ──► [ Signal Processing ] ──► [ AI Classifier ]
                                                      │
                                               [ BCNP Protocol ]
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          ▼                           ▼                           ▼
                    [ Smart Light ]            [ Robot Arm ]              [ Computer ]


System Architecture

BCN consists of two main components:

1. EEG Controller

The brain of the system. Responsible for:


EEG signal acquisition from hardware
Signal preprocessing and feature extraction
AI-based brain signal classification
Sending commands to devices via BCNP


2. BCNP-Compliant Devices

Any device that implements the BCNP interface can be controlled. Examples:


Smart home appliances (lights, fans, TVs)
Robots and actuators
Computers and IoT systems
Assistive technology devices



EEG Paradigms

BCN supports three EEG paradigms to maximize usability across different users:

🏃 Motor Imagery (MI)

Recognition of imagined physical movements (e.g., imagining moving your left hand vs. right hand). Widely studied and well-supported by existing EEG datasets.

🗣️ Imagined Speech (IS)

Recognition of internally spoken words without physical articulation. BCN uses a constrained vocabulary of digits and control words to keep classification tractable.

👁️ Visual Imagery

Mental visualization tasks used as an alternative paradigm for users who find MI or IS difficult.


Design decision: BCN intentionally limits the command vocabulary to a small, well-separated set of mental tasks. A smaller vocabulary significantly improves classification accuracy and makes the system more reliable in practice.




AI Models & Results

All models were implemented in PyTorch and evaluated on EEG datasets for each classification task.

Models Evaluated

ModelTypeCNNSpatial feature extractionCNN + LSTMSpatial + temporal hybridEEGNetGeneral-purpose compact EEG modelDeepConvNetDeep convolutional architectureATCNetAttention-based temporal convolutionBCN ArchitectureCustom architecture developed for this project

Classification Results

Motor Imagery (MI)

MetricValueTaskImagined movement classificationBest Accuracy54%

Digit Classification

MetricValueTaskImagined digit recognitionBest Accuracy55.5%Optimized digit set0, 1, 4, 5 — selected for strongest EEG signal separability

Imagined Speech (IS)

MetricValueTaskInternally spoken word recognitionBest Accuracy63.3%


Note: EEG-based classification is inherently noisy. The accuracies above represent a meaningful signal above chance level for multi-class problems, particularly for a constrained vocabulary operating under real BCI constraints.




Computer Vision Module

BCN integrates a YOLO-based object detection module that allows the system to identify physical devices in the user's environment — combining brain-driven intent with visual perception.

Dataset

DetailInfoClassesTelevision, FanAnnotationManual bounding-box annotation (collected and labeled by the team)ModelYOLOv8 fine-tuned on custom dataset

Future: Device Zero

When the user selects Device 0 via the protocol, the vision module launches automatically to identify the nearest device — eliminating manual registration for known device types.


Installation

bash# Clone the repository
git clone https://github.com/ahmedorapi1/Brain-Control-Network-BCN-
cd Brain-Control-Network-BCN-

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Requirements


Python 3.8+
PyTorch 2.0+
NumPy, SciPy, MNE (EEG processing)
OpenCV, Ultralytics (Computer Vision)
See requirements.txt for the full list



Usage

bash# Run the EEG Controller
python main.py --mode controller

# Run a simulated BCNP device (for testing)
python main.py --mode device --name "SmartLight"

# Run the Computer Vision module standalone
python vision/detect.py --source 0  # 0 = webcam


⚠️ Hardware note: A physical EEG headset is required for live sessions. A simulated signal mode is available for development and testing without hardware.




Project Structure

Brain-Control-Network-BCN-/
│
├── controller/          # EEG acquisition, preprocessing, classification
│   ├── acquisition/     # Hardware interface
│   ├── preprocessing/   # Signal filtering and feature extraction
│   └── models/          # PyTorch EEG classification models
│
├── protocol/            # BCNP implementation
│
├── vision/              # YOLO-based object detection module
│   ├── detect.py
│   ├── dataset/         # Custom TV + Fan dataset
│   └── weights/         # Trained model weights
│
├── devices/             # Example BCNP-compliant device implementations
│
├── notebooks/           # Experiments, model training, and evaluation
│
├── requirements.txt
├── main.py
└── README.md


Contributing

This is a graduation project. The codebase is open for learning and reference.

If you want to build a BCNP-compliant device or extend the AI models, feel free to open an issue or a pull request.


Team

NameRoleAhmed Khaled OrapyAI Models, Computer Vision, Data Pipeline[Teammate name][Role][Teammate name][Role]

Supervised by: [Supervisor name, Mansoura University]


License

This project is licensed under the MIT License — see LICENSE for details.


Brain Control Network — Bridging minds and machines
