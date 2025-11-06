# anomaly_detection_of_AWS_EC2_CPU
Anomaly detection of AWS EC2 CPU and helps to build AIOps projects

# 🚀 EC2 CPU Utilization Anomaly Detection

This project detects anomalies in **AWS EC2 CPU utilization metrics** using two approaches:
- **Isolation Forest (unsupervised)**
- **LSTM Autoencoder (deep learning)**

---

## 🧩 Dataset
Source: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)  
File used: `realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv`

---

## ⚙️ Setup
```bash
git clone https://github.com/yourusername/anomaly-ec2.git
cd anomaly-ec2
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

