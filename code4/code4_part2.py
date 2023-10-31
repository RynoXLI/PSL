import numpy as np
import pandas as pd


def BW_onestep(data: np.ndarray, mx: np.ndarray, mz: np.ndarray,
               w: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Perform one iteration of the Baum-Welch algorithm, improving the estimates of
       the transition probability and emission distribution matrices.

    Args:
        data (np.ndarray): Observations
        mx (int): Count of distinct values X can take
        mz (int): Count of distinct values Z can take
        w (np.ndarray): An mz-by-1 probability vector representing the initial distribution for Z1.
        A (np.ndarray): The mz-by-mz transition probability matrix that
                        models the progression from Zt to Zt+1
        B (np.ndarray): The mz-by-mx emission probability matrix,
                        indicating how X is produced from Z

    Returns:
        (np.ndarray, np.ndarray): Updated A and B matrices
    """
    n = data.shape[0]
    
    # E-step
    # ==========================================================

    # Forward algorithm
    # Alpha is an mz-by-T forward probability matrix
    alpha = np.empty((mz, n))
    # \alpha_1(i) = w(i) B(i, x_1)
    alpha[:, 0] = np.multiply(w, B[:, data[0]])
    for t in range(n - 1):
        # \alpha_{t+1}(i) = \sum_j \alpha_t(j) A(j,i) B(i, x_{t+1})
        alpha[:, t + 1] = (A.T @ alpha[:, t]) * B[:, data[t + 1]]
    
    # Backward algorithm
    # Beta is an mz-by-T backwards probability matrix
    beta = np.empty((mz, n))
    # \beta_n(i) = 1
    beta[:, n - 1] = 1
    for t in np.arange(n - 2, -1, step = -1):
        # \beta{t}(i) = \sum_j A(i, j) B(j, x_{t+1}) \beta_{t+1}(j)
        beta[:, t] = A @ (B[:, data[t + 1]] * beta[:, t + 1])

    # Gamma
    # \gamma_t(i,j) = \alpha_t(i) A(i, j) B(j, x_{t + 1}) \beta_{t+1}(j)
    gamma = np.empty((mz, mz, n - 1))
    for t in range(n - 1):
        for j in range(mz):
            gamma[:, j, t] = alpha[:, t] * A[:, j] * B[j, data[t + 1]] * beta[j, t + 1]

    # M-step
    # ==========================================================

    # Update A
    A = gamma.sum(axis=2) # Sum over time
    A /= A.sum(axis=1).reshape(-1, 1)

    # Update B
    # Marginalized gamma: mz-by-n
    gamma_marginal = np.empty((mz, n))
    # P(Z_t=i \mid x) = \sum_{j=1}^{m_z} P(Z_t=i, Z_{t+1} = j \mid x) = \sum_{j=1}^{m_z} \gamma_t(i j)
    gamma_marginal[:, :n - 1] = gamma.sum(axis=1)
    # P(Z_t=i \mid x) = \sum_{j=1}^{m_z} P(Z_{t-1}=j, Z_t = i \mid x) = \sum_{j=1}^{m_z} \gamma_{t-1}(j, i)
    gamma_marginal[:, n - 1] = gamma[:, :, n - 2].sum(axis=0)
    # B^*(i, l) = \frac{\sum_{t:x_t = l} \gamma_t(i)} {\sum_t \gamma_t(i)}
    for l in range(mx):
        B[:, l] = gamma_marginal[:, data == l].sum(axis=1) / gamma_marginal.sum(axis=1)
    return A, B

def myBW(data: np.ndarray, mx: int, mz: int, w: np.ndarray,
         A: np.ndarray, B: np.ndarray, itmax: int):
    """Perform the Baum-Welch algorithm for the Hidden Markov Model to estimate
       the transition probability and emission distribution matrices.

    Args:
        data (np.ndarray): Observations
        mx (int): Count of distinct values X can take
        mz (int): Count of distinct values Z can take
        w (np.ndarray): An mz-by-1 probability vector representing the initial distribution for Z1.
        A (np.ndarray): The mz-by-mz transition probability matrix that
                        models the progression from Zt to Zt+1
        B (np.ndarray): The mz-by-mx emission probability matrix,
                        indicating how X is produced from Z
        itmax (int): Maximum number of EM step iterations to perform
    """
    # Convert range of X values fron [1, 3] to  [0, 2] to facilitate indexing in Python
    data = data - 1
    for _ in range(itmax):
        A, B = BW_onestep(data, mx, mz, w, A, B)
    return A, B

data = pd.read_csv('coding4_part2_data.txt', header=None).to_numpy().flatten()

# Establish possible observations and number of latent states
mx = np.unique(data).shape[0] # Unique X values
mz = 2 # Given in instructions

# Initialize transition probability and emission distribution matrices
w = np.array((0.5, 0.5))
A = np.full((2, 2), 0.5)
B = np.row_stack([np.array([1, 3, 5]) / 9,
                  np.array([1, 2, 3]) / 6])

# Perform Baum-Welch to find estimates of A and B
A, B = myBW(data, mx, mz, w, A, B, 100)
print(f"A: the {mz}-by-{mz} transition matrix\n\n{A}\n\n"
      f"B: the {mz}-by-{mx} emission matrix\n\n{B}\n")