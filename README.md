# University Campus Security & Student Tracking System

## Overview

A comprehensive AI-powered mass surveillance and security system designed specifically for university campuses to monitor, track, and ensure the safety of students, faculty, and visitors. This system leverages advanced facial recognition technology to provide real-time tracking of student movements across campus facilities, enhancing security measures and creating a safer learning environment.

## 🎯 Key Features

### Real-Time Face Detection & Recognition
- **Advanced MTCNN Integration**: Multi-task Cascaded Convolutional Networks for accurate face detection in various lighting conditions
- **Real-time Processing**: Process live video feeds from multiple campus cameras simultaneously
- **High Accuracy**: Trained models achieving 90%+ recognition accuracy for enrolled students

### Student Movement Tracking
- **Campus-Wide Monitoring**: Track student movements across different campus locations
- **Entry/Exit Logging**: Automated logging of student entry and exit from campus facilities
- **Historical Data**: Maintain comprehensive records of student presence and movement patterns
- **Attendance Verification**: Cross-reference student locations with scheduled classes

### Security & Safety Features
- **Unauthorized Access Detection**: Identify and alert security personnel about unrecognized individuals
- **Emergency Response**: Quickly locate students during emergency situations
- **Suspicious Activity Monitoring**: Flag unusual movement patterns for security review
- **Safe Zone Verification**: Ensure students remain within designated safe areas

### Dual Operation Modes
- **Online Mode**: Cloud-based processing with remote access for security teams
- **Offline Mode**: Local processing for areas with limited connectivity
- **Hybrid Deployment**: Seamlessly switch between online and offline modes

## 🏗️ System Architecture

### Core Components

1. **Face Detection Engine** (`codes/face_finder.py`)
   - Detects faces in video streams and static images
   - Extracts facial features for recognition

2. **Face Recognition Module** (`codes/facenet2.py`)
   - Compares detected faces against enrolled student database
   - Generates confidence scores for matches

3. **Video Processing System** (`codes/video.py`)
   - Processes video feeds from security cameras
   - Handles multiple concurrent video streams

4. **Student Database Management** (`codes/create.py`, `codes/editor.py`)
   - Manages enrolled student facial profiles
   - Updates and maintains student records

5. **Data Consolidation** (`codes/add.py`)
   - Aggregates detection data from multiple sources
   - Generates reports and analytics

## 📊 Data Structure

```
ml-projects/
│
├── codes/                          # Core system modules
│   ├── face_finder.py             # Face detection implementation
│   ├── facenet2.py                # Face recognition engine
│   ├── video.py                   # Video stream processor
│   ├── create.py                  # Student enrollment module
│   ├── editor.py                  # Database editor
│   └── add.py                     # Data consolidation
│
├── volunteers/                     # Enrolled student profiles
│   └── [Student_ID]/              # Individual student directories
│
├── faces/                         # Detected faces from surveillance
│   └── [Student_ID]/              # Matched student faces
│
├── trained_models/                # AI recognition models
│   └── *.pkl                      # Trained model files
│
├── reference_faces/               # Reference images for training
│
└── output/                        # Processing results and logs
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
OpenCV
MTCNN
NumPy
Pillow
Matplotlib
scikit-learn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/palamakulameghanadhgoud/ml.git
cd ml
```

2. **Set up virtual environment**
```bash
python -m venv mlenv
source mlenv/bin/activate  # On Windows: mlenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install opencv-python mtcnn numpy pillow matplotlib scikit-learn
```

4. **Configure camera sources**
- Edit configuration files to add your campus camera feeds
- Set up network access to security cameras

### Quick Start

#### Enrolling Students

```python
# Run the student enrollment module
python codes/create.py

# Follow prompts to:
# - Enter student ID
# - Capture facial images
# - Generate student profile
```

#### Processing Video Feeds

```python
# Start real-time monitoring
python codes/video.py

# Process recorded footage
python codes/face_finder.py --input path/to/video.mp4
```

#### Generating Reports

```python
# Consolidate detection data
python codes/add.py

# View student movement reports in output/
```

## 💼 Use Cases

### 1. **Campus Access Control**
- Monitor entry points to restricted areas
- Verify student authorization for building access
- Track unauthorized access attempts

### 2. **Attendance Management**
- Automated class attendance tracking
- Library and facility usage monitoring
- Event participation verification

### 3. **Emergency Management**
- Rapid student location during emergencies
- Evacuation verification and headcount
- Missing person identification

### 4. **Security Investigations**
- Historical movement data for incident investigations
- Identify individuals present at specific times/locations
- Pattern analysis for security assessments

### 5. **Campus Safety Analytics**
- Identify high-traffic areas requiring additional security
- Monitor late-night campus activity
- Detect unusual behavioral patterns

## 🔒 Privacy & Compliance

### Data Protection
- Encrypted storage of facial biometric data
- Secure transmission protocols for video feeds
- Access controls for authorized personnel only

### Regulatory Compliance
- FERPA compliant student data handling
- GDPR considerations for international students
- Regular privacy impact assessments

### Ethical Guidelines
- Transparent communication with students about surveillance
- Clear policies on data retention and usage
- Regular audits of system usage

## 📈 Performance Metrics

- **Detection Rate**: 95%+ face detection accuracy
- **Recognition Accuracy**: 90%+ match accuracy for enrolled students
- **Processing Speed**: 30+ FPS for real-time video streams
- **False Positive Rate**: <5% with confidence threshold tuning
- **Scalability**: Support for 10,000+ enrolled students

## 🛠️ System Requirements

### Hardware
- **Server**: Multi-core CPU (8+ cores recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 500GB+ for video archives and face database
- **GPU**: CUDA-compatible GPU for accelerated processing (optional)
- **Network**: Gigabit ethernet for camera connectivity

### Cameras
- Minimum 1080p resolution
- RTSP/HTTP stream support
- Fixed position with minimal movement
- Adequate lighting for face detection

## 🔧 Configuration

### Camera Setup
Edit `config.py` to add camera sources:
```python
CAMERA_SOURCES = {
    'main_entrance': 'rtsp://camera1.university.edu/stream',
    'library': 'rtsp://camera2.university.edu/stream',
    'dormitory_a': 'rtsp://camera3.university.edu/stream'
}
```

### Detection Parameters
Adjust sensitivity and thresholds:
```python
CONFIDENCE_THRESHOLD = 0.85  # Match confidence required
DETECTION_INTERVAL = 0.5     # Process every 0.5 seconds
MAX_FACE_SIZE = 1000         # Maximum face detection size
```

## 📱 Deployment Modes

### Online Deployment
- Cloud-based processing on AWS/Azure/GCP
- Remote access for security teams
- Centralized data storage and analytics
- Scalable compute resources

### Offline Deployment
- Local server installation
- Campus network-only access
- On-premise data storage
- Reduced latency for real-time processing

### Hybrid Mode
- Primary online processing with offline backup
- Automatic failover during connectivity issues
- Best of both deployment strategies

## 🎓 Training & Support

### Administrator Training
- System operation and monitoring
- Student enrollment procedures
- Report generation and analysis
- Troubleshooting common issues

### Security Personnel Training
- Alert response procedures
- Investigation workflow
- Privacy policy compliance
- System limitations awareness


## 📜 License

This project is proprietary software intended for university campus security applications. Unauthorized use, distribution, or modification is prohibited.

## ⚠️ Disclaimer

This system is designed as a security tool to enhance campus safety. It should be used in compliance with all applicable laws and regulations regarding surveillance and biometric data collection. Institutions must obtain proper consent and maintain transparency with students regarding the use of this technology.

## 🔄 Future Enhancements

- [ ] Mobile app for security personnel
- [ ] Integration with campus ID card systems
- [ ] Behavioral anomaly detection using AI
- [ ] Multi-campus deployment support
- [ ] Advanced analytics dashboard
- [ ] Integration with emergency alert systems
- [ ] Weapon detection capabilities
- [ ] Crowd density monitoring

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Developed by**: srihan rao, meghanadh goud, yashwanth sai 

