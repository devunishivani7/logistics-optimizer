# Logistics Optimizer

An optimization project for the Capacitated Vehicle Routing Problem (CVRP), comparing CPU and GPU approaches.

## Features

- Genetic Algorithm-based route optimization
- Vehicle capacity constraint validation
- CPU vs GPU benchmark comparison
- Interactive Streamlit dashboard

## Requirements

- Python 3.11
- Git
- An OpenCL-compatible device for GPU execution

## Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd logistics-optimizer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard

```bash
python -m streamlit run dashboard_app.py
```

Open the local URL shown in the terminal.

## Notes

GPU benchmarking requires a compatible OpenCL runtime and device.
If GPU support is unavailable, use the CPU backend where supported.