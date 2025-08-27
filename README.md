# AdversarialAttack

A modular demo of adversarial attacks on pretrained computer vision models using FastAPI (backend) and Streamlit (frontend).

## Structure
- `backend/`: FastAPI app, PyTorch models, attack implementations
- `frontend/`: Streamlit UI
- `download_models.py`: Script to pre-download all models

## Setup

### 1. Install Dependencies
To install all dependencies, run:
```sh
uv sync
```

### 2. Download Models (Recommended)
Pre-download all pretrained models to avoid delays during runtime:
```sh
python download_models.py
```

This script will download:
- ResNet18
- EfficientNet_B0  
- MobileNetV2

The models will be cached locally (~100-200MB total) for faster loading.

### 3. Start the Services

#### Backend
Run FastAPI server:
```sh
uvicorn backend.main:app --reload
```

#### Frontend
Run Streamlit app:
```sh
streamlit run frontend/app.py
```

### 4. Access the Demo
- Open your browser to `http://localhost:8501` for the Streamlit UI
- The FastAPI backend will be running on `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs`

## EC2 Deployment

### Prerequisites
- AWS EC2 instance (t2.large or higher recommended for PyTorch)
- Ubuntu 22.04 LTS
- Security group allowing inbound traffic on ports 8000 and 8501

### 1. Setup EC2 Instance

#### Launch EC2 Instance
```bash
# Recommended instance type: t2.large or t3.large
# AMI: Ubuntu Server 22.04 LTS
# Storage: 20GB minimum (for models and dependencies)
```

#### Configure Security Group
Add inbound rules:
- **Port 8000** (FastAPI): Source = Your IP or 0.0.0.0/0
- **Port 8501** (Streamlit): Source = Your IP or 0.0.0.0/0
- **Port 22** (SSH): Source = Your IP

### 2. Connect to EC2 and Setup Environment

```bash
# Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and pip
sudo apt install python3.11 python3.11-venv python3-pip git -y

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone your project (or upload via scp)
git clone <your-repo-url>
cd AdversarialAttack

# Or upload project files
# scp -i your-key.pem -r AdversarialAttack ubuntu@your-ec2-ip:~/
```

### 3. Install Dependencies and Download Models

```bash
# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Download models (this may take a few minutes)
python download_models.py
```

### 4. Start Services on EC2

#### Option A: Run in Separate Terminals (for testing)

Terminal 1 - Backend:
```bash
# SSH to EC2
ssh -i your-key.pem ubuntu@your-ec2-public-ip
cd AdversarialAttack
source .venv/bin/activate

# Start FastAPI (bind to all interfaces)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 - Frontend:
```bash
# SSH to EC2 (new terminal)
ssh -i your-key.pem ubuntu@your-ec2-public-ip
cd AdversarialAttack
source .venv/bin/activate

# Start Streamlit (bind to all interfaces)
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

#### Option B: Run with Screen (for persistent sessions)

```bash
# Install screen
sudo apt install screen -y

# Start backend in screen session
screen -S backend
cd AdversarialAttack && source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Press Ctrl+A, then D to detach

# Start frontend in screen session
screen -S frontend
cd AdversarialAttack && source .venv/bin/activate
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
# Press Ctrl+A, then D to detach

# List screen sessions
screen -ls

# Reattach to sessions
screen -r backend   # or frontend
```

#### Option C: Run with systemd (for production)

Create service files:

Backend service:
```bash
sudo nano /etc/systemd/system/adversarial-backend.service
```

```ini
[Unit]
Description=Adversarial Attack Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/AdversarialAttack
Environment=PATH=/home/ubuntu/AdversarialAttack/.venv/bin
ExecStart=/home/ubuntu/AdversarialAttack/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Frontend service:
```bash
sudo nano /etc/systemd/system/adversarial-frontend.service
```

```ini
[Unit]
Description=Adversarial Attack Frontend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/AdversarialAttack
Environment=PATH=/home/ubuntu/AdversarialAttack/.venv/bin
ExecStart=/home/ubuntu/AdversarialAttack/.venv/bin/streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable adversarial-backend adversarial-frontend
sudo systemctl start adversarial-backend adversarial-frontend

# Check status
sudo systemctl status adversarial-backend
sudo systemctl status adversarial-frontend
```

### 5. Access the Application Locally

Once services are running on EC2, access from your local machine:

- **Streamlit UI**: `http://your-ec2-public-ip:8501`
- **FastAPI Backend**: `http://your-ec2-public-ip:8000`
- **API Documentation**: `http://your-ec2-public-ip:8000/docs`

### 6. Optional: Setup Reverse Proxy with Nginx

For production deployment with custom domain:

```bash
# Install nginx
sudo apt install nginx -y

# Create nginx config
sudo nano /etc/nginx/sites-available/adversarial-attack
```

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Streamlit frontend
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # FastAPI backend
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/adversarial-attack /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Troubleshooting EC2 Deployment

#### Common Issues:

1. **Connection Refused**
   - Check security group allows ports 8000, 8501
   - Verify services are binding to 0.0.0.0, not localhost

2. **Memory Issues**
   - Use t2.large or higher for PyTorch models
   - Monitor with `htop` or `free -h`

3. **Model Download Fails**
   - Ensure internet connectivity
   - Check disk space with `df -h`

4. **Services Stop**
   - Use screen sessions or systemd for persistence
   - Check logs with `journalctl -u service-name`

#### Useful Commands:

```bash
# Check running processes
ps aux | grep python

# Check port usage
sudo netstat -tlnp | grep :8501
sudo netstat -tlnp | grep :8000

# Monitor resources
htop
df -h
free -h

# Check logs
journalctl -u adversarial-backend -f
journalctl -u adversarial-frontend -f
```

## Usage

1. **Upload an image** using the file uploader
2. **Choose a model** from the sidebar (ResNet18, EfficientNet_B0, MobileNetV2)
3. **Select an attack type** and adjust parameters in the sidebar
4. **Review the attack configuration** displayed below the upload
5. **Click "Launch Attack"** to generate adversarial examples
6. **Compare results** side-by-side to see how the attack affects predictions

## Features
- Choose pretrained model (ResNet18, EfficientNet_B0, MobileNetV2)
- Choose attack type (FGSM, PGD, Gaussian Blur, Salt-and-Pepper Noise, Adversarial Patch)
- Adjust attack parameters dynamically
- View original and adversarial predictions side-by-side
- Detailed probability bar chart comparisons
- Attack success metrics and analysis
- Modern, responsive UI with sidebar controls

## Available Attacks

### Adversarial Attacks
- **FGSM (Fast Gradient Sign Method)**: Single-step gradient-based attack
- **PGD (Projected Gradient Descent)**: Multi-step iterative attack

### Image Corruptions  
- **Gaussian Blur**: Simple image blurring with configurable kernel size
- **Salt & Pepper Noise**: Random pixel corruption with configurable noise level
- **Adversarial Patch**: Overlay attack with a bright square patch

## Technical Details

### Backend (FastAPI)
- RESTful API with `/predict/` and `/attack/` endpoints
- Modular attack implementations in `backend/attacks/`
- Comprehensive error handling and logging
- Input validation and sanitization

### Frontend (Streamlit)
- Modern UI with sidebar controls and responsive layout
- Real-time parameter adjustment based on attack type
- Progress indicators and status feedback
- Interactive charts and metrics display

## Notes
- All attacks implemented in `backend/attacks/`
- Models and utils in `backend/models.py` and `backend/utils.py`
- FastAPI and Streamlit run as separate services
- Uses PyTorch and torchvision for models and attacks
- Pre-downloading models recommended for better performance