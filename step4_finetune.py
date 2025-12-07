import sys
import os
import subprocess

def run_step4():
    print("STEP 4: Joint Fine-Tuning (Denoiser + Classifier)")

    cmd = "python3 src/finetune_and_classify.py"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")
        
    print("\nStep 4 Complete. Fine-tuned model evaluated.")

if __name__ == "__main__":
    run_step4()
