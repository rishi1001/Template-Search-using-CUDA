all:
	nvcc -arch=sm_35 -std=c++11 main.cu -o main
clean:
	rm sample