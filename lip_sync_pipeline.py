import subprocess
import sys
import os

# --- Configuration ---
INPUT_VIDEO = 'input_video.mp4'
NEW_AUDIO = 'new_audio.wav'
OUTPUT_VIDEO = 'output_video.mp4'
WAV2LIP_DIR = 'Wav2Lip'
CHECKPOINT_FILE = os.path.join(WAV2LIP_DIR, 'checkpoints', 'wav2lip_gan.pth')
SFD_MODEL_FILE = os.path.join(WAV2LIP_DIR, 'face_detection', 'detection', 'sfd', 's3fd.pth')
INFERENCE_SCRIPT = os.path.join(WAV2LIP_DIR, 'inference.py')

def run_checks():
    """Performs pre-flight checks to ensure all files and models are ready."""
    print("Step 1: Checking for required files and directories...")
    required_files = [INPUT_VIDEO, NEW_AUDIO]
    for f in required_files:
        if not os.path.isfile(f):
            print(f"--> ERROR: Required file not found: {f}")
            return False
    if not os.path.isdir(WAV2LIP_DIR):
        print(f"--> ERROR: Wav2Lip directory not found: {WAV2LIP_DIR}")
        return False
    print("--> All required files and directories are present.\n")

    print("Step 2: Checking for pre-trained models...")
    model_files = [CHECKPOINT_FILE, SFD_MODEL_FILE]
    for m in model_files:
        if not os.path.isfile(m):
            print(f"--> ERROR: Pre-trained model not found: {m}")
            print("--> Please ensure you have downloaded the models and placed them in the correct folders as per the README.")
            return False
    print("--> Pre-trained models are in place.\n")
    return True

def main():
    """Main function to execute the Wav2Lip pipeline."""
    if not run_checks():
        sys.exit(1)

    print("Step 3: All checks passed. Starting Wav2Lip inference process...")
    print("This may take some time depending on your video length and hardware.")

    # --- CRITICAL FIX: Convert all paths to absolute paths ---
    checkpoint_path_abs = os.path.abspath(CHECKPOINT_FILE)
    face_video_abs = os.path.abspath(INPUT_VIDEO)
    audio_file_abs = os.path.abspath(NEW_AUDIO)
    output_file_abs = os.path.abspath(OUTPUT_VIDEO)
    inference_script_abs = os.path.abspath(INFERENCE_SCRIPT)
    
    # Get the python executable from the current virtual environment
    python_executable = sys.executable

    command = [
        python_executable,
        inference_script_abs,
        '--checkpoint_path', checkpoint_path_abs,
        '--face', face_video_abs,
        '--audio', audio_file_abs,
        '--outfile', output_file_abs,
        '--resize_factor', '2'  # Reduces memory usage
    ]

    try:
        # We need to run the command from within the Wav2Lip directory
        # for it to find its internal helper files (`audio.py`, etc.)
        subprocess.run(command, check=True, cwd=WAV2LIP_DIR)
        print("\n----------------------------------------------------")
        print(f"✅ Success! Your lip-synced video has been saved as: {OUTPUT_VIDEO}")
        print("----------------------------------------------------")
    except subprocess.CalledProcessError as e:
        print("\n----------------------------------------------------")
        print(f"❌ An error occurred while running the Wav2Lip inference script: {e}")
        print("--> Please check the output above for specific error messages from the script.")
        print("----------------------------------------------------")
        sys.exit(1)

if __name__ == '__main__':
    main()

