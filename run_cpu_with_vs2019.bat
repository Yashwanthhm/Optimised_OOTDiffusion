@echo off
echo ========================================
echo OOTDiffusion CPU Runner with VS 2019
echo ========================================
echo.

REM Initialize VS 2019 Build Tools environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Activate conda environment (robustly via conda.bat)
call "C:\Users\HP\anaconda3\condabin\conda.bat" activate ootd
if errorlevel 1 (
    echo Conda environment 'ootd' not found. Please create it with: conda create -n ootd python=3.10
    pause
    exit /b 1
)

REM Set CUDA architecture (even for CPU, some extensions may need it)
set TORCH_CUDA_ARCH_LIST=8.6

REM Clear any cached builds
echo Clearing cached CUDA extensions...
if exist "%USERPROFILE%\.cache\torch_extensions\py310_cu121\inplace_abn" (
    rmdir /s /q "%USERPROFILE%\.cache\torch_extensions\py310_cu121\inplace_abn"
    echo Cache cleared.
) else (
    echo No cache found.
)
echo.

REM Run the CPU script
echo Starting OOTDiffusion (CPU mode)...
echo Note: Running on CPU - this will be slower than GPU
cd /d D:\OOTDiffusion\run
python run_ootd_cpu_fixed.py --model_path ../images/model.jpeg --cloth_path ../images/cloth.jpeg --model_type dc --category 2 --scale 2.0 --sample 1

pause
