#include <pnl/pnl_random.h>
#include <cmath>
using namespace std;
int main()
{
    ///Algorithm parameres///
    int M = 10000;       // M -- the number of simulations
    int N = 1; // N -- the number of time steps
    double dt;// -- time step
    ///Kou model parameters///
    double lam = 3.;
    double P = 0.6;
    double Lp = 15;
    double Lm = 10;
    double sigma = 0.3;
    double gam0 = 0.2;
    //Asset parameters//
    double h = 0.15; // X-loss level
    //VaR and iVaR parameters//
    double gamma = 0.99;  // confidence level
    double T = 10. / 252.; // time horizon (1 business day=1/252 od business year)

    int k; //the number of the current sample path
    int j; //the number of jumps
    int i; //the number of the current jump
    int t; // the number of time step
    int cross = 0; // the number of crossing events;

    int generator;//random generator
    int index; // VaR index
    double VaR;  //VaR(X)
    double iVaR; //iVaR(X)
    double w;//loss level hitting risk

    PnlRng* rng; //random number generator
    PnlVect* X = pnl_vect_create_from_zero(M); // vector of samples of the process position
    PnlVect* MX = pnl_vect_create_from_zero(M); // vector of samples of the supremum process position
    //Choose and create a generator

    generator = PNL_RNG_MERSENNE;
    rng = pnl_rng_create(generator);
    /* rng must be initialized. */
    pnl_rng_sseed(rng, 0);
    /////Example of sampling from the standard normal distribution

    printf("Sample from the standard normal distribution: %.4f\n", pnl_rng_normal(rng));
    /////Example of sampling from the uniform distribution

    printf("Sample from the uniform distribution[0,1]:   %.4f\n", pnl_rng_uni(rng));
    /////Example of sampling from the Poisson law with the parameter lambda=3.00

    printf("Sample from the Poisson law with the parameter lambda=%.2f: %i\n", lam, pnl_rng_poisson(lam,rng));
    /////Example of sampling from the double exponential law

    printf("Sample from the double exponential law with Lambda_+=%.2f, Lambda_-=%.2f, P=%.2f: %.4f\n", Lp, Lm, P, pnl_rng_dblexp(Lp,Lm,P,rng));


    /////////////////////Computing loss level hitting risk over horizon///////////////
    ///////////////without partition of the time period//////////////////
    N = 1;
    cross = 0;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    {   /////sampling from Kou model
        LET(X, k) = sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt;
        j = pnl_rng_poisson(lam*dt, rng); // draw a sample from the Poisson law to compute the number of jumps
        for (i = 0; i < j; i++)
        {
            LET(X, k) = GET(X, k) + pnl_rng_dblexp(Lp, Lm, P, rng); // draw a sample from asymmetric double exponential distribution to compute the size of the jump and add it to X[k]
        }
        ////end sampling

        //check if the barrier is crossed
        if (GET(X, k) > h) { cross = cross + 1; };
    }
    //estimate the risk
    w = (double) cross / M;
    printf("\n risk to hit loss level at %.2f is equal to %.8f\n", h, w);




    ///////////////with partition of the time period//////////////////
    N = 10;
    cross = 0;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    {   /////sampling from Kou model
        LET(X, k) = 0;
        for (t = 0; t < N; t++)
        {
            LET(X, k) = GET(X, k) + sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt;
            j = pnl_rng_poisson(lam * dt, rng); // draw a sample from the Poisson law to compute the number of jumps
                for (i = 0; i < j; i++)
                {
                    LET(X, k) = GET(X, k) + pnl_rng_dblexp(Lp, Lm, P, rng);// draw a sample from asymmetric double exponential distribution to compute the size of the jump to compute the size of the jump and add it to X[k]
                }
        //check if the barrier is crossed
            if (GET(X, k) > h) { cross = cross + 1; };
        }////end sampling
    }
    //estimate the risk
    w = (double)cross / M;     
    printf(" risk to hit loss level at %.2f is equal to %.8f\n", h, w);
    /////////////////////end - Computing loss level hitting risk/////////////////////


    /////////////////////VaR(X) computation/////////////////////
    //// Sort the vector X using PNL according to increasing order
    pnl_vect_qsort(X, 'i');

    /// Find the index corresponding and print it
    index = (int)(gamma * M) - 1;
    printf("Value-at-Risk index= %i\n", index);
    /// Find and print it

    VaR = GET(X, index);
    printf("VaR(X) = %.15f at confidence level %.2f\n", VaR, gamma);
    /////////////////end-VaR(X) computation/////////////////////


    //////////////////// iVaR(X) computation/////////////////////  
    N = 10;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    {

        /////sampling from Kou model
        LET(X, k) = 0;
        for (t = 0; t < N; t++)
        {
            LET(X, k) = GET(X, k) + sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt; // draw a sample from the Gaussian part of X_{ dt } ^ k and add it to X[k]
            j = pnl_rng_poisson(lam * dt, rng);  // draw a sample from the Poisson law to compute the number of jumps
            for (i = 0; i < j; i++)
            {
                LET(X, k) = GET(X, k) + pnl_rng_dblexp(Lp, Lm, P, rng); // draw a sample from asymmetric double exponential distribution to compute the size of the jump and add it to X[k]
                               
            } ////end sampling

            // simulate the supremum process and put it into vector MX
            if (GET(X, k) > GET(MX, k)) LET(MX, k) = GET(X, k);
        }
    }


    //// Sort the vector MX using PNL according to increasing order
    pnl_vect_qsort(MX, 'i');
            
    /// Find the index corresponding and print it
    index = (int)(gamma * M) - 1;
    printf("intra horizon Value-at-Risk index= %i\n", index);
        
    /// Find  and print it
    iVaR = GET(MX, index);
    printf("iVaR(X) = %.15f at confidence level %.2f\n", iVaR, gamma);


   /////////////////////end-iVaR(X) computation/////////////////////


    pnl_vect_free(&X);        // Frees the pnl vector //

    pnl_vect_free(&MX);        // Frees the pnl vector //

    pnl_rng_free(&rng);      // Frees the generator //

    exit(0);

}
