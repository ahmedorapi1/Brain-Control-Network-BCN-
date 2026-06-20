# 🧠 Brain Control Network (BCN)

> Control multiple network-connected devices using only your brain activity.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Protocol: BCNP](https://img.shields.io/badge/Protocol-BCNP-brightgreen.svg)](#bcnp-protocol)
[![Paradigms: MI · IS · Visual](https://img.shields.io/badge/EEG-MI%20%7C%20IS%20%7C%20Visual-purple.svg)](#brain-signal-representation)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Brain Signal Representation](#brain-signal-representation)
- [BCNP Protocol](#bcnp-protocol)
- [Protocol Workflow](#protocol-workflow)
- [AI Models](#ai-models)
- [Requirements](#requirements)
- [Main Contributions](#main-contributions)

---

## Overview

**Brain Control Network (BCN)** is an EEG-based Brain-Computer Interface (BCI) system that enables users to control multiple network-connected devices using only brain activity.

Unlike traditional EEG control systems that are limited to a single device and require device-specific software, BCN introduces a standardized communication protocol — **BCNP (Brain Control Network Protocol)** — that allows dynamic discovery, configuration, and control of an arbitrary number of devices.

---

## Problem Statement

Most existing EEG-controlled systems suffer from critical limitations:

| Limitation | Description |
|---|---|
| **Single-device control** | Most systems can only control one device at a time |
| **Device-specific configuration** | Every new device requires dedicated software and manual setup |
| **Static mappings** | Brain commands must be configured in advance and cannot be changed dynamically |
| **Lack of scalability** | No standardized method for discovering and managing multiple devices simultaneously |

BCN was designed to solve all of these problems at the protocol level.

---

## Objectives

BCN was designed around three primary goals:

### 1. Zero Pre-Configuration
Devices should be discoverable and usable without installing custom software for each device.

### 2. Multi-Device Support
A single EEG controller should be able to communicate with and control an arbitrary number of devices simultaneously.

### 3. Dynamic Command Mapping
Users should be able to register, modify, and remove brain-command mappings **during runtime** without restarting the system.

---

## System Architecture

The BCN ecosystem consists of two major components:

```
┌─────────────────────────────────┐        BCNP         ┌──────────────────────┐
│         EEG Controller          │ ◄─────────────────► │  BCNP-Compliant      │
│                                 │                      │  Devices             │
│  • EEG Acquisition Hardware     │                      │                      │
│  • Signal Preprocessing         │                      │  • Smart Home        │
│  • Brain Signal Classifier      │                      │  • Robots            │
│  • BCNP Protocol Stack          │                      │  • Computers         │
└─────────────────────────────────┘                      │  • IoT Systems       │
                                                         │  • Assistive Tech    │
                                                         └──────────────────────┘
```

### EEG Controller
The controller translates brain activity into BCNP commands and sends them to network devices. It handles EEG acquisition, signal preprocessing, and AI-based classification.

### BCNP-Compliant Devices
Any device implementing BCNP can join the network and expose its functionalities to the controller. The protocol abstracts internal device details — the controller does not need to know how a device works internally.

---

## Brain Signal Representation

BCN intentionally limits the command space to improve EEG classification accuracy and reliability.

### Supported Symbols

| Category | Symbols |
|---|---|
| **Digits** | `0` `1` `2` `3` `4` `5` `6` `7` `8` `9` |
| **Control Words** | `Help` · `Hello` · `Register` · `Forget` · `End` · `Network` · `Device` |

> A small, constrained vocabulary significantly improves EEG classification accuracy and usability.

### Supported EEG Paradigms

- **Motor Imagery (MI)** — Recognition of imagined movements
- **Imagined Speech (IS)** — Recognition of internally spoken words
- **Visual Imagery** — Visual mental tasks

Different mental tasks can map to the same protocol symbol, improving robustness and usability.

---

## BCNP Protocol

**BCNP (Brain Control Network Protocol)** is a custom binary application-layer protocol specifically designed for EEG-based control systems.

### Core Features

- Device discovery
- Device registration
- Network management
- Action registration
- Dynamic remapping
- Session management

### Command Hierarchy

Commands are organized in a three-level hierarchy that enables large-scale device management:

```
Network Key
      │
      └── Device Key
                │
                └── Action Key
```

This structure allows the system to scale to large numbers of devices across multiple networks.

---

## Protocol Workflow

### 1. Device Discovery
```
Controller ──[Discover Request]──► Network
Network    ──[Discover Announce]──► Controller
```
The controller broadcasts a discovery request. Responding devices send a `Discover Announce`. The controller builds a table of available devices.

### 2. Device Registration
Each discovered device receives a **Network Key** and a **Device Key**, stored for future sessions.

### 3. Session Establishment
```
Controller ──[Hello]──► Device
Device     ──[Hello Pair]──► Controller
```
A temporary paired session is created for device configuration and action discovery.

### 4. Action Discovery
The controller requests available actions from the device:

```
Examples:
  • Turn light on / off
  • Open / close door
  • Move robot forward / backward
  • Play / pause music
```

### 5. Dynamic Mapping (Runtime)
Brain commands are mapped to device actions at runtime:

```
Digit 1  →  Turn on lamp
Digit 2  →  Open garage door
Digit 3  →  Move robot forward
```

Mappings can be **modified at any time** without restarting the system.

### 6. Forget Mechanism
Previously registered mappings can be removed **individually** or **entirely** via the `Forget` control word.

### 7. Session Termination
```
Controller ──[End]──► Device
Device     ──[End Bye]──► Controller
```
The device returns to idle mode after session termination.

---

## AI Models

BCN investigated several deep learning architectures for EEG signal classification:

| Model | Description |
|---|---|
| **CNN** | Convolutional Neural Network for spatial EEG feature extraction |
| **CNN + LSTM** | Hybrid model combining spatial and temporal learning |
| **EEGNet** | Compact, general-purpose EEG classification model |
| **DeepConvNet** | Deep convolutional architecture for EEG |
| **ATCNet** | Attention-based Temporal Convolutional Network |
| **Proposed Architecture** | Enhanced custom architecture developed for BCN |

### Classification Tasks

- **Motor Imagery Classification** — Recognizing imagined movements
- **Digit Classification** — Classifying digits used in BCNP commands
- **Imagined Speech Classification** — Recognizing internally spoken words
- **Computer Vision Integration** — YOLO-based object detection for environmental perception

---

## Requirements

### Functional Requirements

- EEG signal acquisition
- Brain signal classification
- Command generation
- Device discovery & selection
- Network management
- Command execution
- Session management
- Dynamic action mapping
- Multi-device support

### Non-Functional Requirements

| Quality | Description |
|---|---|
| **Accuracy** | Small vocabulary improves EEG classification performance |
| **Reliability** | Commands and acknowledgments use reliable communication |
| **Scalability** | Supports arbitrary numbers of devices and networks |
| **Low Latency** | Near real-time response for interactive control |
| **Usability** | Simple commands that users can memorize |
| **Maintainability** | Modular protocol architecture |
| **Interoperability** | Any hardware platform can participate if it implements BCNP |

---

## Main Contributions

1. **BCNP** — A novel application-layer protocol for EEG-based multi-device control
2. **Scalable Hierarchy** — A Network → Device → Action command architecture
3. **Dynamic Registration** — Runtime registration and remapping of device actions
4. **Constrained Vocabulary** — A human-manageable, EEG-optimized command set
5. **Standardized Framework** — A unified approach to EEG-based device discovery and communication
6. **Device-Agnostic Architecture** — Control of heterogeneous systems without pre-configuration

---

## Conclusion

Brain Control Network proposes a new paradigm for EEG-based human-computer interaction — moving beyond single-device control and introducing a **protocol-level solution** for multi-device environments.

By combining brain-computer interfaces, artificial intelligence, network protocols, dynamic device discovery, and runtime action mapping, BCN establishes the foundation for a scalable ecosystem where users can seamlessly control multiple devices using only their brain activity.

This opens new possibilities for:
- 🏥 Assistive technologies
- 🏠 Smart home environments
- 🤖 Robotics
- 💻 Next-generation human-computer interaction

---

*Brain Control Network — Bridging minds and machines through protocol.*
