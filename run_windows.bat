@echo off
set INPUT=data\input
set OUTPUT=output
set ALPHA=1.2
set BETA=20
set STREAMS=4

REM Add OpenCV + CUDA DLLs to PATH for this session
set PATH=%PATH%;C:\Users\Anya\Downloads\opencv\build\x64\vc16\bin;"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"

if not exist bin\batch_proc.exe (
    echo Build first: make
    pause
    exit /b 1
)

bin\batch_proc.exe %INPUT% %OUTPUT% %ALPHA% %BETA% %STREAMS%
echo Done. Logs in %OUTPUT%\timings.csv
pause
