# AI-Driven Time-Series Anomaly Detection

**Intelligent Time-Series Anomaly Detection for Server and Application Monitoring**

This project integrates deep learning models directly into an existing observability pipeline to provide intelligent, real-time anomaly detection for infrastructure metrics. By combining **Prometheus**, **Grafana**, and **OpenTelemetry** with **AI models (LSTM Autoencoder, TCN, and Transformer)**, the system learns normal server behavior and instantly flags unexpected traffic spikes, latency anomalies, or resource exhaustion.

---

## 🚀 Key Features

- **Deep Learning Engine:** Features three hot-swappable AI models tailored for time-series data:
  - `LSTM Autoencoder`: Detects anomalies using reconstruction error.
  - `Temporal Convolutional Network (TCN)`: Uses dilated causal convolutions to capture long-range trends.
  - `Transformer`: Leverages self-attention to spot complex temporal irregularities.
- **Real-Time Data Pipeline:** A FastAPI backend seamlessly queries live metrics directly from Prometheus using PromQL.
- **Dynamic Thresholding:** Automatically calculates `warning` and `critical` severity thresholds based on historical model errors (mean + std deviation analysis).
- **Premium Web Dashboard:** A standalone, dark-themed UI built with HTML/JS/CSS and Chart.js, providing interactive graphs, model status tracking, and a real-time anomaly alerts panel.
- **OpenTelemetry Microservices Demo:** Built on top of a fully-functional 10+ microservice eCommerce mock environment, complete with a load generator to simulate traffic spikes.
- **Dockerized Ecosystem:** The entire stack (Backend, Frontend, Prometheus, Grafana, ML Service) spins up effortlessly with Docker Compose.

---

## 🏗️ Architecture

1.  **Metric Collection:** Node Exporter and OpenTelemetry microservices expose metrics, which are continually scraped by **Prometheus**.
2.  **AI Backend (FastAPI):** Continually queries `request rates`, `CPU usage`, `memory`, and `latency` from Prometheus.
3.  **Data Preprocessing:** Performs dynamic sliding-window segmentation, handling missing values, and min-max normalization.
4.  **Inference Engine:** Feeds sequences into the active deep learning model (PyTorch CPU-optimized).
5.  **Dashboard:** Reacts via polling to display anomaly scores and trigger UI alerts when scores break calculated dynamic thresholds.

---

## 🛠️ Quick Start Guide

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- (Optional but recommended) At least 8GB RAM allocated to Docker.

### 1. Build and Run the Stack

Clone the repository and spin up the environment:

```bash
cd demo
# Clean old containers
docker compose --env-file .env down --remove-orphans

# Build the AI anomaly detector
docker compose --env-file .env build anomaly-detector

# Start the entire ecosystem
docker compose --env-file .env up --force-recreate --remove-orphans --detach
```

_(Note: On subsequent runs, you only need to execute `docker compose --env-file .env up -d`)_

### 2. Access the Application Services

Once the containers are successfully running, the following portals are available locally:

| Service                     | Access URL                          | Description                                                           |
| :-------------------------- | :---------------------------------- | :-------------------------------------------------------------------- |
| **🧠 AI Anomaly Dashboard** | `http://localhost:8501`             | **Main interface!** View live metrics, AI status, and anomaly alerts. |
| **🛍️ OpenTelemetry App**    | `http://localhost:8082`             | The mock webstore being monitored.                                    |
| **🔥 Load Generator**       | `http://localhost:8082/loadgen/`    | Tool for simulating traffic spikes to trigger anomalies.              |
| **📊 Grafana**              | `http://localhost:8082/grafana/`    | Baseline PromQL visualizations.                                       |
| **📈 Prometheus**           | `http://localhost:8082/prometheus/` | Raw metric data and target health.                                    |

### 3. Running the Demo / Testing Anomalies

1.  Open the **AI Anomaly Dashboard** (`http://localhost:8501`).
2.  Click the **"🧠 Train Models"** button in the top right. The system will pull recent history and learn the environment's baseline.
3.  Open the **Load Generator** (`http://localhost:8082/loadgen/`) and exponentially increase the simulated users (e.g., from 10 to 100).
4.  Switch back to the AI Dashboard. Within 15-30 seconds, you will observe the anomaly score drastically spike and turn Red, generating a timestamped alert in the panel.

---

## 📂 Project Structure

```bash
.
├── ai-anomaly-detector/       # AI FastAPI Backend
│   ├── models/                # PyTorch Archs: lstm_autoencoder, tcn, transformer, anomaly_engine
│   ├── dashboard/             # Custom HTML/JS Frontend UI
│   ├── main.py                # FastAPI routing and entry point
│   ├── preprocessing.py       # Time-series segmentation and normalization
│   ├── prometheus_client.py   # HTTP client handling PromQL queries
│   ├── requirements.txt       # Python dependencies (CPU-only PyTorch)
│   └── Dockerfile             # Container instructions for the AI service
├── demo/
│   ├── docker-compose.yml     # Orchestrates Prometheus, Grafana, Microservices, + our AI Service
│   ├── .env                   # Port and container config (AI port mapping: 8501)
├── rules/                     # Example native PromQL baseline alerting rules
└── README.md                  # This file
```

---

## 📜 Credits & Provenance

- **EDI Project Team:** Built by [Your Name/Team] for Vishwakarma Institute of Technology (VIT) - Spring 2026.
- **Infrastructure Baseline:** The core microservice tracking, node-exporter, Prometheus, and Grafana wiring was originally inspired by / adapted from the `grafana/promql-anomaly-detection` repository. All advanced machine learning engines, dynamic thresholding, data pipelining, API backend, and the interactive anomaly dashboard have been uniquely architected and implemented specifically for this project.
