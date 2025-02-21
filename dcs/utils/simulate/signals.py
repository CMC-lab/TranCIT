import numpy as np


def generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2, apply_morlet=False):
    X = np.zeros((2, T - 500, Ntrial))
    
    if apply_morlet == True:
        ns_x = 0.02 * np.concatenate([
            np.ones(650), 
            np.ones(201) - morlet(-0.29, 0.29, 201), 
            np.ones(150)
        ])
    else:
        ns_x = 0.02 * np.ones(T + 1)
        
    ns_y = 0.005 * np.ones(T + 1)
    
    for N in range(Ntrial):
        x = np.random.rand(2)
        y = np.random.rand(2)
        
        c2 = 0
        c1 = 0.098

        for t in range(1, T - 1):
            x = np.append(x, (2 - gamma1*h) * x[-1] + (-1 + gamma1*h - h**2 * Omega1**2) * x[-2] + h**2 * ns_x[t] * np.random.randn() + h**2 * c2 * y[-2])
            y = np.append(y, (2 - gamma2*h) * y[-1] + (-1 + gamma2*h - h**2 * Omega2**2) * y[-2] + h**2 * ns_y[t] * np.random.randn() + h**2 * c1 * x[-2])

        u = np.array([x[500:], y[500:]])
        X[:, :, N] = u
    
    return X, ns_x, ns_y

def morlet(start, end, num_points):
    t = np.linspace(start, end, num_points)
    w0 = 5
    sigma = 1.0
    return np.cos(w0 * t) * np.exp(-t**2 / (2 * sigma**2))


    # x = np.zeros(T + 1)
    # y = np.zeros(T + 1)
    # x[:2] = np.random.rand(2)  # Random initial values
    # y[:2] = np.random.rand(2)  # Random initial values
        
    # Loop to update x and y over time
    # for t in range(1, T-1):
    #     # Update x and y using the given equations
    #     x[t+1] = (2 - gamma1 * h) * x[t] + (-1 + gamma1 * h - h**2 * Omega1**2) * x[t-1] + h**2 * ns_x[t] * np.random.randn() + h**2 * c2 * y[t-1]
    #     y[t+1] = (2 - gamma2 * h) * y[t] + (-1 + gamma2 * h - h**2 * Omega2**2) * y[t-1] + h**2 * ns_y[t] * np.random.randn() + h**2 * c1 * x[t-1]
        
    # Extract the relevant part of x and y (after t=500)
    # u = np.array([x[501:], y[501:]])
    # u = np.array([x[1:], y[1:]])
    # X[:, :, N] = u  # Store in X array
