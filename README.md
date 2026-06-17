# # Physical Activity Recognition & Fall Detection via Wearable Sensors

> **BSc Computer Science Final Year Project**  
> **Academic Institution:** University of East Anglia (UEA)  
> **Academic Classification:** First-Class Honours (Grade: **73.75%**)  
> **Core Architecture:** Modular Python Signal Processing & Machine Learning Pipeline

---

## 📈 Project Overview & Key Achievements
This project designs, implements, and critically evaluates a production-grade machine learning and deep learning pipeline to recognize human activities and detect falls using high-frequency wearable sensor telemetry from the **SisFall dataset**. 

By challenging the modern assumption that model scale always outpaces deliberate engineering, this framework evaluates traditional classifiers augmented with a custom 112-channel time-frequency domain feature extraction pipeline against an end-to-end Long Short-Term Memory (LSTM) recurrent neural network.

### 📊 Performance Benchmarks
Through rigorous validation utilizing SMOTE class-balancing and 5-fold stratified cross-validation, the optimized, feature-engineered K-Nearest Neighbours (KNN) model emerged as the definitive champion:

| Model Architecture | Average Accuracy | Fall Class F1-Score | Compute Latency / Resource Cost |
| :--- | :--- | :--- | :--- |
| **Feature-Engineered KNN** | **99.54%** | **0.993** | **Ultra-Low (Real-time Deployment Ready)** |
| End-to-End LSTM Network | 96.77% | 0.942 | High (Compute Intensive) |
| Random Forest | 98.10% | 0.975 | Moderate |
| Support Vector Machine (SVM) | 97.45% | 0.961 | Moderate |

> 💡 **Core Engineering Insight:** Thoughtful, domain-specific feature design—rather than raw neural network complexity—is the decisive variable in maximizing predictive accuracy while minimizing execution latency for edge-device deployments.

---

## 📂 Repository Architecture
The codebase is modularized according to industry-standard software engineering principles to ensure separation of concerns, scalability, and seamless integration with external creative tools:

```text
FYP_ActivityRecognition/
│
├── data/                  # Local storage directories (Excluded from version control)
│   ├── SisFall_dataset/   # Raw acceleration and gyroscopic data streams (.txt)
│   ├── features/          # Extracted feature matrices (.csv / .npy)
│   └── results/           # Performance charts, confusion matrices, and plots
│
├── notebooks/             # Exploratory Data Analysis (EDA) and prototyping scratchpads
│
├── src/                   # Production-grade source pipeline scripts
│   ├── __init__.py
│   ├── data_preprocessing.py   # Butterworth filtering and window segmentation
│   ├── feature_engineering.py  # 112-channel time/frequency domain math engines
│   ├── model_training.py       # Cross-validation loops and SMOTE augmentation
│   └── evaluate.py             # Statistical narratives and reporting generators
│
├── .gitignore             # Airtight data-clutter exclusion boundaries
├── README.md              # Executive project documentation
└── requirements.txt       # Verified environment execution dependencies