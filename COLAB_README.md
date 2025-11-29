# Google Colab Integration - Summary

This repository has been modified to work seamlessly in Google Colab with GPU or TPU support.

## New Files Created

### 1. `check_colab_environment.py`
- **Purpose**: Comprehensive environment check script
- **Features**:
  - Detects if running in Google Colab
  - Checks GPU/TPU availability
  - Verifies Python version
  - Checks installed packages
  - Verifies directory structure
  - Checks data files
  - Reports disk space

**Usage**:
```bash
python check_colab_environment.py
```

### 2. `setup_colab.py`
- **Purpose**: Automated dependency installation for Colab
- **Features**:
  - Installs PyTorch with CUDA support
  - Installs all required packages
  - Verifies installation
  - Optional TPU support installation

**Usage**:
```bash
python setup_colab.py
```

### 3. `train_colab.py`
- **Purpose**: Colab-compatible training script
- **Key Modifications**:
  - Automatic Colab detection
  - Reduced `num_workers` (Colab-friendly)
  - Optional Neptune logging (can be disabled)
  - TPU support
  - Better error handling
  - Graceful degradation if dependencies missing

**Usage**:
```bash
# Basic usage
python train_colab.py -C cfg_1 --fold 0

# With options
python train_colab.py -C cfg_1 --fold 0 --use_tpu --disable_neptune
```

### 4. `run_colab.py`
- **Purpose**: Interactive pipeline runner
- **Features**:
  - Step-by-step execution
  - User prompts for configuration
  - Error handling
  - Progress tracking

**Usage**:
```bash
python run_colab.py
```

### 5. `requirements_colab.txt`
- **Purpose**: Colab-specific requirements file
- **Features**:
  - Compatible package versions
  - Optional TPU libraries (commented)
  - Clear documentation

### 6. `COLAB_SETUP.md`
- **Purpose**: Comprehensive setup guide
- **Contents**:
  - Quick start guide
  - Step-by-step instructions
  - GPU/TPU setup
  - Troubleshooting
  - Configuration tips

### 7. `colab_notebook_template.ipynb`
- **Purpose**: Ready-to-use Jupyter notebook
- **Features**:
  - Pre-configured cells
  - Step-by-step execution
  - All training steps included
  - Google Drive integration

## Key Modifications

### Automatic Adjustments for Colab
1. **num_workers**: Automatically reduced to 2 (from 8) for Colab compatibility
2. **pin_memory**: Set to False to avoid Colab issues
3. **Neptune**: Made optional with graceful fallback
4. **Device detection**: Automatic GPU/TPU/CPU detection

### Command Line Interface
- Added `--fold` argument for easier fold specification
- Added `--use_tpu` flag for TPU training
- Added `--disable_neptune` flag to skip logging

## Quick Start in Colab

### Method 1: Using the Notebook
1. Open `colab_notebook_template.ipynb` in Google Colab
2. Run cells sequentially
3. Follow the prompts

### Method 2: Using Scripts
```bash
# 1. Check environment
python check_colab_environment.py

# 2. Setup dependencies
python setup_colab.py

# 3. Run training
python train_colab.py -C cfg_1 --fold 0
```

### Method 3: Interactive Pipeline
```bash
python run_colab.py
```

## Workflow

The training workflow remains the same as the original:

1. **Round 1**: Train 4 folds of cfg_1
   ```bash
   python train_colab.py -C cfg_1 --fold 0
   python train_colab.py -C cfg_1 --fold 1
   python train_colab.py -C cfg_1 --fold 2
   python train_colab.py -C cfg_1 --fold 3
   ```

2. **Generate OOF**: Create train_folded_oof_supp.csv
   ```bash
   python scripts/get_train_folded_oof_supp.py
   ```

3. **Round 2**: Train cfg_2 with fullfit
   ```bash
   python train_colab.py -C cfg_2 --fold -1
   python train_colab.py -C cfg_2 --fold -1  # Second seed
   ```

4. **Convert to TF-Lite**
   ```bash
   python scripts/convert_cfg_2_to_tf_lite.py
   ```

## Differences from Original

| Feature | Original | Colab Version |
|---------|----------|---------------|
| num_workers | 8 | 2 (auto-adjusted) |
| pin_memory | Configurable | False (auto-adjusted) |
| Neptune | Required | Optional |
| TPU Support | No | Yes (optional) |
| Error Handling | Basic | Enhanced |
| Environment Check | Manual | Automated |

## Troubleshooting

### Common Issues

1. **CUDA not available**
   - Enable GPU in Colab: Runtime → Change runtime type → GPU

2. **Out of Memory**
   - Reduce batch_size in config
   - Use cfg_1 instead of cfg_2
   - Enable mixed_precision

3. **Neptune errors**
   - Use `--disable_neptune` flag
   - Or set up Neptune API token

4. **Data not found**
   - Download using Kaggle API
   - See COLAB_SETUP.md for instructions

## Notes

- All original functionality is preserved
- Original `train.py` remains unchanged
- Colab-specific files are clearly named
- Backward compatible with local execution
- Works with both GPU and TPU

## Support

For detailed instructions, see:
- `COLAB_SETUP.md` - Complete setup guide
- `README.md` - Original repository documentation
- `colab_notebook_template.ipynb` - Interactive notebook

