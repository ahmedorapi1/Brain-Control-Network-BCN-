# Brain Control Network (BCN)

BCN is an EEG-based Brain-Computer Interface system that enables users to control multiple network-connected devices simultaneously using only brain activity.

The system introduces BCNP, a custom application-layer protocol that allows any compliant device to join the network and be controlled dynamically, without device-specific software or manual pre-configuration.

---

## System Architecture

BCN consists of two main components.

The **EEG Controller** handles signal acquisition from the headset, preprocessing, AI-based classification, and sending commands to devices over the network.

**BCNP-Compliant Devices** are any devices that implement the BCNP interface — smart home appliances, robots, computers, or IoT systems. The controller does not need to know how a device works internally; it only communicates through the protocol.

---

## EEG Paradigms

BCN supports three EEG paradigms.

**Motor Imagery (MI)** — recognition of imagined physical movements such as imagining moving the left or right hand.

**Imagined Speech (IS)** — recognition of internally spoken words without physical articulation. BCN uses a constrained vocabulary of digits and control words to keep classification reliable.

**Visual Imagery** — mental visualization tasks used as an alternative for users who find MI or IS more difficult.

The command vocabulary is intentionally kept small. A smaller set of well-separated mental tasks significantly improves classification accuracy and makes the system more practical.

---

## AI Models and Results

All models were implemented in PyTorch and evaluated on EEG datasets for each classification task.

Models evaluated include CNN, CNN + LSTM, EEGNet, DeepConvNet, ATCNet, and a custom architecture developed specifically for BCN.

**Motor Imagery** — best accuracy: 54%

**Digit Classification** — best accuracy: 55.5%. The digit set was reduced from 10 to 4 digits (0, 1, 4, 5), selected because they produce the strongest and most distinguishable EEG signals.

**Imagined Speech** — best accuracy: 63.3%

---

## Computer Vision Module

BCN integrates a YOLO-based object detection module that allows the system to identify physical devices in the user's environment, combining brain-driven intent with visual perception.

The dataset was collected and annotated manually by the team, covering two classes: Television and Fan. These were chosen as representative smart home devices to validate the vision pipeline as a proof of concept.

A planned future feature called Device Zero will allow the user to select device 0 via the protocol, which automatically launches the vision module to identify the nearest device — removing the need for manual registration.

---

## Team
Mohamed Salah El-Din – Team Leader and responsible for the communication protocol.

Ahmed Khaled Oraby – Responsible for AI models, Computer Vision, and the Data Pipeline.

Ahmed Osama – Responsible for Hardware Development.

Karam Youssef – Responsible for the Image-to-Speech Model.

Omar Ali – Responsible for Data Collection



