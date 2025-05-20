all:
	g++ allocator.cpp -o allocator.so -I/opt/cuda/include -shared -fPIC
