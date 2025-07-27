CUDA_PATH := "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
NVCC      := $(CUDA_PATH)/bin/nvcc.exe

# Use the VS2019-compatible v142 toolset under VS2022
CCBIN := "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64"

ARCH_FLAGS = -gencode arch=compute_50,code=sm_50

NVCCFLAGS = -ccbin $(CCBIN) -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler="/MD"

OPENCV_INC = C:/Users/Anya/Downloads/opencv/build/include
OPENCV_LIB = C:/Users/Anya/Downloads/opencv/build/x64/vc16/lib

CXXFLAGS = -O2 -std=c++17 -I$(OPENCV_INC)
LDFLAGS  = -L$(OPENCV_LIB) -lopencv_world4110 -lcudart

SRC = src/main.cu
BIN = bin/batch_proc.exe

all: $(BIN)

$(BIN): $(SRC)
	@if not exist bin mkdir bin
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(CXXFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

clean:
	del /Q bin\*.exe 2> NUL || exit 0
	del /Q output\*.* 2> NUL || exit 0
