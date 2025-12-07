import sys
import os
import subprocess

sys.path.append(os.getcwd())

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def run_step3():
    print("STEP 3: JOINT Pre-train on Unlabeled Data (Arxiv Only)")

    print("\n[3.1] Training GNN Encoder + Diffusion Denoiser Jointly...")
    run_command("python3 src/train_joint_pretrain.py")
    
    print("\nStep 3 Complete. Jointly trained models saved.")

if __name__ == "__main__":
    run_step3()
