@echo off
echo ========================================
echo OOTDiffusion - 4GB GPU Optimized
echo ========================================
echo.

REM Initialize Visual Studio 2019 Build Tools environment
echo Initializing Visual Studio 2019 Build Tools...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
echo.

REM Activate conda environment (robustly via conda.bat)
call "C:\Users\HP\anaconda3\condabin\conda.bat" activate ootd
if errorlevel 1 (
    echo Conda environment 'ootd' not found. Please create it with: conda create -n ootd python=3.10
    pause
    exit /b 1
)

REM Set environment variables for aggressive memory optimization
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_NO_CUDA_MEMORY_CACHING=0

REM Set CUDA architecture (adjust based on your GPU)
set TORCH_CUDA_ARCH_LIST=8.6

REM Clear CUDA cache
echo Clearing CUDA cache...
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
echo.

REM Run the low VRAM script
echo Starting OOTDiffusion (4GB GPU mode - FAST 10min target)...
echo Note: Optimized for speed - reduced steps for faster processing
echo Target completion time: ~10 minutes
echo.
cd /d D:\OOTDiffusion\run
python run_ootd_low_vram.py --model_path ../images/model.jpeg --cloth_path ../images/cloth.jpeg --model_type dc --category 2 --scale 1.8 --sample 1 --step 10

pause
