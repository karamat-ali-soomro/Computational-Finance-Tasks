import numpy as np

# Algorithm parameters
M = 10000 # M -- the number of simulations
N = 50     # N -- the number of time steps
dt = 0.0  # time step

# Kou model parameters
lam = 5.0
mu = 0.2
delta = 0.4
sigma = 0.3 # volatility
gam0 = 0.2 # drift

# Asset parameters
h = 0.15 # X-loss level    

# VaR and iVaR parameters
gamma = 0.99 # confidence level
T = 20 / 252 # time horizon (1 business day = 1/252 of a business year)

# k -- the number of the current sample path
# j -- the number of jumps
# i -- the number of the current jump
# t -- the number of time step

# index -- VaR index
# VaR -- VaR(X)
# iVaR -- iVaR(X)
# w -- loss level hitting risk

# Initialize generator
rng = np.random.default_rng(seed=0)

# Generate samples
X = np.zeros(M)
MX = np.zeros(M)


####### Computing loss level hitting risk over horizon #######
########### without partition of the time period #############

N = 1
cross = 0
dt = T / N # time step setup

for k in range(0, M):

    # Sampling from Kou model
    X[k] = 0.0

    # Generate the number of jumps
    j = rng.poisson(lam * dt)

    # Simulate jumps
    for i in range(j):
            Y = rng.normal(mu, delta)
            X[k] = X[k] + Y

    # Add the Gaussian part and drift
    X[k] = X[k] + sigma * np.sqrt(dt) * rng.normal(0, 1) + gam0 * dt

    # Check if the barrier is crossed
    if X[k] >= h:
        cross = cross + 1

# Estimate the risk
w = cross / M

print("Without partition of the time period")
print(f"Risk to hit loss level at {h:.2f} is equal to {w:.8f}\n")


############ with partition of the time period ############

N = 50
cross = 0
dt = T / N # time step setup

for k in range(0, M):

    # Sampling from Kou model
    X[k] = 0.0
    hit = False

    for t in range(0, N):

        # Generate the number of jumps
        j = rng.poisson(lam * dt)

        # Simulate jumps and accumulate them
        for i in range(j):
            Y = rng.normal(mu, delta)
            X[k] = X[k] + Y

            # Check if the barrier is crossed after each jump
            if X[k] >= h:
                hit = True

        # Add the Gaussian part and drift
        X[k] = X[k] + sigma * np.sqrt(dt) * rng.normal(0, 1) + gam0 * dt

        # Check if the barrier is crossed after the time-step increment
        if X[k] >= h:
            hit = True

    # Count paths that crossed the barrier
    if hit:
        cross = cross + 1

# Estimate the risk
w = cross / M

print("With partition of the time period")
print(f"Risk to hit loss level at {h:.2f} is equal to {w:.8f}\n")


############# end - Computing loss level hitting risk #################


# ================ VaR(X) computation ======================

# Simulate terminal losses over the time horizon using N = 10
N = 10
dt = T / N

for k in range(0, M):

    X[k] = 0.0

    for t in range(0, N):

        # Sampling from Kou model
        j = rng.poisson(lam * dt)

        # Accumulate jumps
        for i in range(j):
            Y = rng.normal(mu, delta)
            X[k] = X[k] + Y

        # Add the Gaussian part and drift
        X[k] = X[k] + sigma * np.sqrt(dt) * rng.normal(0, 1) + gam0 * dt


# Sort the vector X in increasing order
X_sort = np.sort(X)

# Find the index corresponding to the confidence level
index = int(gamma * M) - 1

print(f"Value-at-Risk index = {index}")

# Find and print VaR
VaR = X_sort[index]
print(f"VaR(X) = {VaR:.15f} at confidence level {gamma:.2f}\n")

################## end-VaR(X) computation ###################


#################### iVaR(X) computation ####################

N = 50
dt = T / N # time step setup

for k in range(0, M):

    # Sampling from Kou model
    X[k] = 0.0
    MX[k] = 0.0

    for t in range(0, N):

        # Generate the number of jumps
        j = rng.poisson(lam * dt)

        # Accumulate jumps and update the supremum process
        for i in range(j):
            Y = rng.normal(mu, delta)
            X[k] = X[k] + Y

            # Update the supremum process after each jump
            if X[k] > MX[k]:
                MX[k] = X[k]

        # Add the Gaussian part and drift
        X[k] = X[k] + sigma * np.sqrt(dt) * rng.normal(0, 1) + gam0 * dt

        # Update the supremum process after the time-step increment
        if X[k] > MX[k]:
            MX[k] = X[k]


# Sort the vector MX in increasing order
MX_sort = np.sort(MX)

# Find the index corresponding to iVaR_gamma(X_T)
index = int(gamma * M) - 1

print(f"Intra-horizon Value-at-Risk index = {index}")

# Find and print iVaR
iVaR = MX_sort[index]
print(f"iVaR(X) = {iVaR:.15f} at confidence level {gamma:.2f}")

############### end-iVaR(X) computation ####################
import matplotlib.pyplot as plt
x = np.linspace(0,1,M)
plt.plot(x,X)
plt.show()
