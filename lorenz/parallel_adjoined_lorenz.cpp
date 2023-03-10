#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <chrono>

#include "ad.hpp"
#include <mpi.h>

auto init_time(){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    static auto init = Time::now();
    return init;
}

template<class T>
double time_it(T&& fun, std::string what = ""){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    const auto init = init_time();

    const auto t0_f = Time::now();
    fun();
    const auto t1_f = Time::now();
    fsec diff = t1_f - t0_f;
    if(what!=""){
        //std::cout << what << " " << diff.count() << " " << fsec(t1_f-init).count() << std::endl;
    }
    return diff.count();
}

template<class T>
void partial_path(const int from, const int to, const std::vector<T>& x_in, std::vector<T>& x_out, const std::vector<T>& p, const double dt){
    static std::vector<T> x = x_in;
    const T& beta  = p[0];
    const T& sigma = p[1];
    const T& rho   = p[2];

    for(int it=from; it<to; it++){
        x = {
            x[0] + dt*sigma*(x[1]-x[0]),
            x[1] + dt*(x[0]*(rho-x[2])-x[1]),
            x[2] + dt*(x[0]*x[1]-beta*x[2])
        };
    }
    x_out = x;
}

int main(int c, char* v[]) {
    MPI_Init(&c, &v);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int I = world_rank;
    const int N = world_size;

    using AFLT_T = ad::adjoint_t<double>; // active type
    using PFLT_T = double; // passive type

    std::vector<std::string> timing_strings = {
        "passiveForward",
        "activeForward",
        "allocate",
        "waitForAdjoint",
        "interpret",
        "reduce"
        };
    std::vector<std::vector<double>> timings(N,std::vector<double>(timing_strings.size(),0.0));

    PFLT_T dt=0.001;
    int n_steps = 40/dt;

    std::vector<AFLT_T> x = {1,1,1};
    std::vector<AFLT_T> p = {8.0/3.0, 10.0, 28.0};

    std::string myProc = "_" + std::to_string(world_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    time_it([&](){
        ad::mode<AFLT_T>::global_tape = ad::mode<AFLT_T>::tape_t::create(/*opts*/);
    }, "createTape" + myProc);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    const auto t1 = Time::now();

    ad::mode<AFLT_T>::global_tape->register_variable(p);

    int passive_from = 0;
    int active_from = int(std::floor(1.0*I*n_steps/N));
    int passive_to = active_from;
    int active_to = std::min(int(std::ceil(1.0*(I+1)*n_steps/N)),n_steps);

    //std::vector<double> xP_in = ad::value(x), xP_out, pP = ad::value(p);
    std::vector<double> xP_in(x.size()), xP_out, pP(p.size());
    for(int i=0; i<x.size(); i++){
        xP_in[i] = ad::value(x[i]);
    }
    for(int i=0; i<p.size(); i++){
        pP[i] = ad::value(p[i]);
    }
    
    std::cout << "Proc " <<  I << ": passive from " << passive_from << " to " << passive_to << std::endl;
    timings[I][0] = time_it([&](){
        partial_path(passive_from,passive_to,xP_in,xP_out,pP,dt);
    }, "passiveForward" + myProc);
  
    std::vector<AFLT_T> x_in(xP_out.begin(),xP_out.end()), x_out;
    ad::mode<AFLT_T>::global_tape->register_variable(p);
    ad::mode<AFLT_T>::global_tape->register_variable(x_in);
  
    std::cout << "Proc " <<  I << ": active from " << active_from << " to " << active_to << std::endl;
    timings[I][1] = time_it([&](){
        partial_path(active_from,active_to,x_in,x_out,p,dt);
    }, "activeForward" + myProc);
  
    std::cout << "x_out: " << x_out[0] << " " << x_out[1] << " " << x_out[2] << std::endl;

    std::vector<double> seed = {1.0,0.0,0.0};
    fsec time_mpi;
  
    timings[I][2] = time_it([&](){
            ad::derivative(x_out[0]) = seed[0]; // preallocate adjoint vector
            ad::derivative(x_out[1]) = seed[1]; 
            ad::derivative(x_out[2]) = seed[2]; 
    },"allocate" + myProc);
    if(I<N-1){
        timings[I][3] = time_it([&](){
            MPI_Recv(/*buf*/ seed.data(), /*count*/ 3, /*datatype*/ MPI_DOUBLE, /*dest*/ world_rank+1, /*tag*/ 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }, "mpiRecv" + myProc);
    }

    timings[I][4] = time_it([&](){
        ad::derivative(x_out[0]) = seed[0];
        ad::derivative(x_out[1]) = seed[1];
        ad::derivative(x_out[2]) = seed[2];
        ad::mode<AFLT_T>::global_tape->interpret_adjoint();
    }, "interpret" + myProc);

    if(I>0){
        time_it([&](){
            std::vector<double> adjoint = 
                {ad::derivative(x_in[0]), ad::derivative(x_in[1]), ad::derivative(x_in[2])};
            MPI_Send(/*buf*/ adjoint.data(), /*count*/ 3, /*datatype*/ MPI_DOUBLE, /*dest*/ I-1, /*tag*/ 0, MPI_COMM_WORLD);
        },"mpiSend" + myProc);
    }

    std::vector<double> dxdp = {ad::derivative(p[0]), ad::derivative(p[1]), ad::derivative(p[2])};
    std::vector<double> dxdp_sum = dxdp;
  
    timings[I][5] = time_it([&](){
        //MPI_reduce(&dxdp, &dxdp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(dxdp.data(), dxdp_sum.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }, "mpiReduce" + myProc);

    if(I==0){
        std::cout << "dxdp " << dxdp[0] << " " << dxdp[1] << " " << dxdp[2] << std::endl;
        std::cout << "dxdp " << ad::derivative(x_in[0]) << " " 
                             << ad::derivative(x_in[1]) << " " 
                             << ad::derivative(x_in[2]) << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fsec wallclock = Time::now() - t1;

    if(I==0){
        std::cout << "wallclock: " << wallclock.count() << "ms" << std::endl;
        for(int i=1; i<N; i++){
            MPI_Recv(
                /*buf*/ timings[i].data(),
                /*count*/ timings[i].size(),
                /*datatype*/ MPI_DOUBLE,
                /*src*/ i, /*tag*/ 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }

        std::cout << "Processor";
        for(int i=0; i<timing_strings.size(); i++){
            std::cout << " " << timing_strings[i];
        }
        std::cout << std::endl;
        for(int i=0; i<N; i++){
            if(N>1)
                std::cout << "P" << i;
            else
                std::cout << "Serial";

            for(int j=0; j<timing_strings.size(); j++){
                std::cout << " " << timings[i][j];
            }
            std::cout << std::endl;
        }
    }else{
        MPI_Send(
            /*buf*/ timings[I].data(),
            /*count*/ timings[I].size(),
            /*datatype*/ MPI_DOUBLE,
            /*dest*/ 0, /*tag*/ 0, MPI_COMM_WORLD
        );
    }

    if(I==0){
        std::cout << std::endl;
        for(int i=0; i<timing_strings.size(); i++){
            std::cout << timing_strings[i] << " ";
            for(int j=0; j<N; j++){
                std::cout << " " << timings[j][i];
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
