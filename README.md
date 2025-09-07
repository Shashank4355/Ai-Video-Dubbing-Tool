# AI Lip Sync Pipeline (Wav2Lip Implementation)

This project provides an end-to-end pipeline to synchronize the lip movements of a person in a video with a new, external audio track. It uses the powerful Wav2Lip model to generate high-fidelity results while maintaining the subject's identity and natural head movements from the original video.

The core of this repository is the `lip_sync_pipeline.py` script, which acts as a robust wrapper to automate the complex inference process, manage system resources, and prevent common file path errors.

## Features

* **High-Quality Lip Sync:** Leverages the Wav2Lip generative model for accurate and realistic lip shape generation based on audio input.
* **Automated Pipeline Script:** Simplifies execution by handling complex command-line arguments and file paths automatically.
* **Environment Validation:** Checks for required files and models before starting, preventing errors mid-process.
* **Memory Management:** Built-in resize factor helps reduce RAM/VRAM usage, making it possible to run on systems with limited resources.
* **Path Correction:** Automatically resolves relative paths to prevent common file I/O errors, especially on Windows operating systems.

## Technology Stack

* **Core Model:** Wav2Lip
* **Face Detection:** S3FD (Single Shot Scale-invariant Face Detector)
* **Framework:** PyTorch
* **Libraries:** OpenCV, Librosa, NumPy

---

## Setup and Installation

Follow these steps carefully to set up the project environment and install all necessary components.

### Step 1: Download Project and Set Up Environment

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Install Python:** This project requires **Python 3.8**. Ensure it is installed on your system.
3.  **Create Virtual Environment:** Open a terminal in the root project directory (`Lip_sync_final/`) and create a virtual environment.

    ```bash
    # Windows command
    py -3.8 -m venv Wav2Lip/wav2lip_env
    ```

4.  **Activate Environment:**

    ```bash
    # Windows command prompt
    .\Wav2Lip\wav2lip_env\Scripts\activate
    ```

### Step 2: Download Pre-trained Models

The pipeline requires two pre-trained models. You must download these from the official Wav2Lip model sources.

* **Wav2Lip GAN Model:** Download `wav2lip_gan.pth`.
* **S3FD Face Detector Model:** Download `s3fd.pth`.


# Step 3: Install Dependencies

1.  **Modify `requirements.txt`:** The standard Wav2Lip requirements can have compatibility issues. Open `Wav2Lip/requirements.txt` and replace its contents with the following specific versions:

    ```plaintext
    librosa==0.7.0
    numpy==1.17.3
    torch==1.8.0
    torchvision==0.9.0
    tqdm==4.45.0
    numba==0.48
    ```

2.  **Install Requirements:** With your virtual environment active, navigate into the `Wav2Lip` directory and run:

    ```bash
    # Ensure you are in the Wav2Lip directory for this step
    cd Wav2Lip
    pip install -r requirements.txt
    cd ..
    ```

3.  **Install Specific OpenCV Version:** Install the required version of OpenCV separately.

    ```bash
    pip install opencv-python==4.5.5.64
    ```

---

## How to Run the Pipeline

1.  **Prepare Input Files:**
    * Place your source video in the root folder and rename it to `input_video.mp4`.
    * Place your new audio track in the same folder and rename it to `new_audio.wav`.

2.  **Execute Script:**
    * Ensure your virtual environment (`wav2lip_env`) is active.
    * From the root project directory (`Lip_sync_final/`), run the main pipeline script:

    ```bash
    python lip_sync_pipeline.py
    ```

The script will display progress in the terminal. Once complete, the final video will be saved as `output_video.mp4` in your project root directory.

---

## Performance Note

This process is computationally intensive. Running on a system with an NVIDIA GPU (CUDA enabled) is highly recommended for significantly faster processing. The script includes a memory-saving resize factor by default, but execution on a CPU may still take a very long time for longer videos.
Place the downloaded models into the correct directories as shown below.
```plaintext
Lip_sync_final/
└── Wav2Lip/
    ├── checkpoints/
    │   └── wav2lip_gan.pth   <-- Place main model here
    │
    └── face_detection/
        └── detection/
            └── sfd/
                └── s3fd.pth  <-- Place face detector here
