all: passive_lorenz serial_adjoined_lorenz parallel_adjoined_lorenz

passive_lorenz:
	g++ -O3 passive_lorenz.cpp -o passive_lorenz.out

serial_adjoined_lorenz:
	g++ -O3 serial_adjoined_lorenz.cpp -o serial_adjoined_lorenz.out

parallel_adjoined_lorenz:
	mpic++ -O3 parallel_adjoined_lorenz.cpp -o parallel_adjoined_lorenz.out

run_serial: serial_adjoined_lorenz
	./lorenz.out | tee lorenz.dat

run_parallel: parallel_adjoined_lorenz
	mpirun -np 4 parallel_adjoined_lorenz.out

clean:
	rm -f *.out