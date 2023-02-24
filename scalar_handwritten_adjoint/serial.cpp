#include <iostream>
#include <stack>
#include <cmath> // sin,cos
#include <tuple>

double f(double x, const double p){ 
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
    for(int i=0; i<n; i++)
        x = f(x,p);
    return x;
}

std::tuple<double,double,double> adjoint_evolution(int n, double x0, const double p){
    std::stack<double> fds;
    double x = x0;
    // augmented forward
    for(int i=0; i<n; i++){
        fds.push(x);
        x = f(x,p);
    }
    double y = x;
    
    // reverse
    double a1_p = 0, a1_x = 1.0;
    for(int i=n-1; i>=0; i--){
        x = fds.top(); fds.pop();
        a1_p += dfdp(x,p)*a1_x; // increment of adjoint parameter
        a1_x  = dfdx(x,p)*a1_x; // no increment due to aliasing
    }
    return {y, a1_x, a1_p}; // dx/dx0, dx/dp
}

int main(){
    double x = 1.0, p = 1.0;
    auto [y, dydx, dydp] = adjoint_evolution(10, x, p);
    
    std::cout << "y = "    <<    y << std::endl;
    std::cout << "dydx = " << dydx << std::endl;
    std::cout << "dydp = " << dydp << std::endl;
}