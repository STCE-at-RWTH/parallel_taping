all: serial parallel

serial:
	g++ -O3 serial.cpp -o serial.out

parallel:
	mpic++ -O3 parallel.cpp -o parallel.out

clean:
	rm -f serial.out parallel.out
