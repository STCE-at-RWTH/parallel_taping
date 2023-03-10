#include <iostream>
#include <array>

#include "ad.hpp"

int main(){
    using ad_mode = ad::adjoint<double>;
    using ad_type = ad_mode::type;

    ad_mode::global_tape = ad_mode::tape_t::create();

    double dt = 0.001;

    std::array<ad_type,3> x = {1,1,1};
    ad_type sigma = 10, rho = 28, beta = 8.0/3.0;

    // register state
    ad_mode::global_tape->register_variable(x[0]);
    ad_mode::global_tape->register_variable(x[1]);
    ad_mode::global_tape->register_variable(x[2]);

    // register parameters
    ad_mode::global_tape->register_variable(sigma);
    ad_mode::global_tape->register_variable(rho);
    ad_mode::global_tape->register_variable(beta);


    long it=0;
    for(double t=0; t<40; t+=dt, it++){
        x = {
            x[0] + dt*sigma*(x[1]-x[0]),
            x[1] + dt*(x[0]*(rho-x[2])-x[1]),
            x[2] + dt*(x[0]*x[1]-beta*x[2])
        };
        
        if(it%10==0){
            std::cout << t << " " << x[0] << " " << x[1] << " " << x[2] << std::endl;
        }
    }

    // reduce result to a scalar state (distance from origin)
    ad_type J = sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2],2));
    
    // seed output
    ad::derivative(J) = 1.0;

    ad_mode::global_tape->interpret_adjoint();

    std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
    std::cout << "dJ/dSigma: " << ad::derivative(sigma) << std::endl;
    std::cout << "dJ/dRho: "   << ad::derivative(rho)   << std::endl;
    std::cout << "dJ/dBeta: "  << ad::derivative(beta)  << std::endl;

    // cleanup
    ad_mode::tape_t::remove(ad_mode::global_tape);
}