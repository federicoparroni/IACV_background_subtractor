#include <iostream>
using namespace std;

class PBAS
{
    private:
        int N;
        int K;
        float R_incdec;
        int R_lower;
        int R_scale;
        float T_dec;
        int T_inc;
        int T_lower;
        int T_upper;

    public:
        PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200);
        ~PBAS();
};

PBAS::PBAS(int N, int K=2, float R_incdec=0.05, int R_lower=18, int R_scale=5, float T_dec=0.05, int T_inc=1, int T_lower=2, int T_upper=200)
{
    cout << 'Ciao!\n';
}

PBAS::~PBAS()
{
}
