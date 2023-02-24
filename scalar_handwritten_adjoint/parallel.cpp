#include <iostream>
#include <cmath>
#include <tuple>
#include <mpi.h>

int I, N;

double    f(double x, const double p){
    return p*sin(x);
}

double dfdp(double x, const double p){
    return sin(x);
}

double dfdx(double x, const double p){
    return p*cos(x);
}

double passive_evolution(int n, double x0, const double p){
    double x = x0;
    for(int i=0; i<n; i++){
        x = f(x,p);
    }
    return x;
}

// parallel adjoint evolution
std::tuple<double,double,double> adjoint_evolution(int n, double x0, const double p){
    // calculate I steps in passive mode
    double x = x0;
    x = passive_evolution(I,x,p);
    // calculate one I+1th step augmented forward
    double y = f(x,p);

    // reverse
    double a1_p, global_a1_p, a1_x;
    if(I==N-1){
        // we are at the end of the line, initialize adjoints, seed output
        a1_x = 1.0;
        a1_p = 0;
    }else{
        // get adjoint of state x from next processor
        MPI_Recv(&a1_x, 1, MPI_DOUBLE, I+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // reversal of one step
    a1_p += dfdp(x,p)*a1_x; // increment of adjoint parameter
    a1_x  = dfdx(x,p)*a1_x; // no increment due to aliasing

    if(I>0){
        MPI_Send(&a1_x, 1, MPI_DOUBLE, I-1, 0, MPI_COMM_WORLD);
    }

    MPI_Allreduce(&a1_p, &global_a1_p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {y, a1_x, global_a1_p}; // dx/dx0, dx/dp
}

int main(int c, char* v[]) {
    MPI_Init(&c, &v);
    MPI_Comm_rank(MPI_COMM_WORLD, &I);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    double x = 1.0;
    double p = 1.0;
    auto [y, dydx, dydp] = adjoint_evolution(N, x, p);

    if(I==N-1){
        std::cout << "y = " << y << std::endl;
    }
    if(I==0){
        std::cout << "dydx = " << dydx << std::endl;
        std::cout << "dydp = " << dydp << std::endl;
    }

    MPI_Finalize();
    return 0;
}
