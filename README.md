# Rocket Tracker

## Project Setup

### Prerequisites
- Python 3.8+
- Inference
- RPi.GPIO
# Hardware
- Raspberry Pi 4 or 5
- Picamera module

### Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/brian-io/rocket-detect.git
cd rocket-tracker
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source ./.venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Environment
- Copy `.env.example` to `.env`
- Edit `.env` with your specific configurations

### Running the Application
```bash
python main.py
```

### Note
- Ensure you have a valid Roboflow API key
- Verify the video source path in the `.env` file