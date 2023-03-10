#include <iostream>
#include <array>

int main(){
    std::array<double,3> x = {1,1,1};
    double dt = 0.001;
    double sigma = 10, rho = 28, beta = 8.0/3.0;
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
}