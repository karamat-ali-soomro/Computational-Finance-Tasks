#include <pnl/pnl_random.h>
#include <cmath>
using namespace std;
int main()
{
    ///Algorithm parameters///
    int M = 10000; // M -- the number of simulations
    int N = 1; // N -- the number of time steps
    double dt;// -- time step
    ///Gaussian Levy model parameters///
    double sigma = 0.3; //volatility
    double gam0 = 0.2; //drift
    //Asset parameters//
    double h = 0.15; // X-loss level
    //VaR and iVaR parameters//
    double gamma = 0.99; // confidence level
    double T = 10. / 252.; // time horizon (1 business day=1/252 od business year)

    int k; //the number of the current sample path
    int t; // the number of time step
    int cross = 0; // the number of crossing events;
    int generator;//random generator
    int index; // VaR index
    double VaR; //VaR(X)
    double iVaR; //iVaR(X)
    double w;//loss level hitting risk

    PnlRng* rng; //random number generator
    PnlVect* X = pnl_vect_create_from_zero(M); // vector of samples of the process position
    PnlVect* MX = pnl_vect_create_from_zero(M); // vector of samples of the supremum process position

    generator = PNL_RNG_MERSENNE;
    rng = pnl_rng_create(generator);
    /* rng must be initialized. */
    pnl_rng_sseed(rng, 0);

    /////Example of sampling from the standard normal distribution
    printf("Sample from the standard normal distribution: %.4f\n", pnl_rng_normal(rng));
    ///Example of sampling from the uniform distribution. Insert your code to simulate uniform disribution using a correspondent PNL function
    printf("Sample from the uniform distribution[0,1]:   %.4f\n", pnl_rng_uni(rng));

    /////////////////////Computing loss level hitting risk over horizon///////////////
    ///////////////without partition of the time period//////////////////
    N = 1;
    cross = 0;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    { /////sampling from Gaussian Levy model
        LET(X, k) = sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt;
        ////end sampling

       //check if the barrier is crossed
        if (GET(X, k) > h) { cross = cross + 1; }
    }
    //estimate the risk

    w = (double)cross / M;

    printf(" risk to hit loss level at %.2f is equal to %.8f\n", h, w);


    ///////////////with partition of the time period//////////////////
    N = 10;
    cross = 0;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    { /////sampling from Gaussian Levy model
        LET(X, k) = 0;
        for (t = 0; t < N; t++)
        {
            LET(X, k) = GET(X, k) + sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt;
        }////end sampling 
       //check if the barrier is crossed
        if (GET(X, k) > h) { cross = cross + 1; }
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




    //////////////////// iVaR(X) computation /////////////////////  
    N = 10;
    dt = T / N; /// time step setup
    for (k = 0; k < M; k++)
    {
        /////sampling supremum from Gaussian Levy model
        LET(X, k) = 0;
        LET(MX, k) = 0;                            // <-- initialize running max
        for (t = 0; t < N; t++)
        {
            LET(X, k) = GET(X, k) + sigma * sqrt(dt) * pnl_rng_normal(rng) + gam0 * dt;
            // draw a sample from the Gaussian part and add it to X[k]

            // simulate the supremum process and put it into vector MX
            if (GET(X, k) > GET(MX, k)) LET(MX, k) = GET(X, k);   // <-- update running max
        } ////end sampling
    }

    //// Sort the vector MX using PNL according to increasing order
    pnl_vect_qsort(MX, 'i');                       // <-- sort MX ascending

    /// Find the index corresponding iVaR_gamma(X_T) and print it
    index = (int)(gamma * M) - 1;                  // <-- 0-based index
    printf("intra horizon Value-at-Risk index= %i\n", index);

    /// Find iVaR_gamma(X_T) and print it
    iVaR = GET(MX, index);
    printf("iVaR(X) = %.15f at confidence level %.2f\n", iVaR, gamma);


   /////////////////////end-iVaR(X) computation/////////////////////

    pnl_vect_free(&X);        // Frees the pnl vector //
    pnl_vect_free(&MX);        // Frees the pnl vector //
    pnl_rng_free(&rng);      // Frees the generator //
    exit(0);
}
