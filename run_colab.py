"""
Main execution script for Google Colab
This script orchestrates the entire training pipeline in Colab
"""

import os
import sys
import subprocess

def run_step(description, command, check=True):
    """Run a step and handle errors"""
    print("\n" + "="*60)
    print(description)
    print("="*60)
    print(f"Command: {command}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0 and check:
        print(f"❌ Error in: {description}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ Completed: {description}")
    
    return result.returncode == 0

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("ASL FINGERSPELLING - COLAB TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Check environment
    print("\n📋 Step 1: Checking environment...")
    run_step(
        "Environment Check",
        "python check_colab_environment.py",
        check=False
    )
    
    # Step 2: Setup (optional - ask user)
    print("\n📦 Step 2: Setup dependencies...")
    response = input("Install/update dependencies? (y/n): ")
    if response.lower() == 'y':
        run_step(
            "Installing Dependencies",
            "python setup_colab.py"
        )
    
    # Step 3: Check data
    print("\n📊 Step 3: Checking data...")
    if not os.path.exists("datamount/train_folded.csv"):
        print("⚠️  Data files not found!")
        print("You need to download the data first.")
        print("\nOptions:")
        print("1. Use Kaggle API to download")
        print("2. Upload data manually to datamount/ directory")
        response = input("\nContinue with training anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nPlease download the data first. See README.md for instructions.")
            return
    
    # Step 4: Training configuration
    print("\n🚀 Step 4: Training Configuration")
    print("\nAvailable configs:")
    print("  - cfg_1: Round 1 training (smaller model)")
    print("  - cfg_2: Round 2 training (larger model)")
    
    config = input("\nEnter config name (default: cfg_1): ").strip() or "cfg_1"
    fold = input("Enter fold number (0-3, or -1 for fullfit, default: 0): ").strip() or "0"
    
    # Step 5: Run training
    print("\n🏋️  Step 5: Starting training...")
    cmd = f"python train_colab.py -C {config} --fold {fold}"
    
    use_tpu = input("Use TPU? (y/n, default: n): ").strip().lower() == 'y'
    if use_tpu:
        cmd += " --use_tpu"
    
    disable_neptune = input("Disable Neptune logging? (y/n, default: n): ").strip().lower() == 'y'
    if disable_neptune:
        cmd += " --disable_neptune"
    
    print(f"\nStarting training with command:")
    print(f"  {cmd}")
    print("\n" + "="*60)
    
    run_step(
        "Training",
        cmd,
        check=False  # Don't exit on training errors, let user see the output
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check output in datamount/weights/")
    print("2. For round 2, run: python scripts/get_train_folded_oof_supp.py")
    print("3. Then train cfg_2: python train_colab.py -C cfg_2 --fold -1")

if __name__ == "__main__":
    main()

