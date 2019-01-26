#include <iostream>

#include "pbas.h"

using namespace std;


PBAS::PBAS()
{
    cout << "Ciao!" << endl;
}
PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200)
{
    cout << "Ciao!" << endl;
}

PBAS::~PBAS()
{
}

int main() {
    PBAS *p = new PBAS();
    
    return 0;
}
