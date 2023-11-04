import numpy as np
import pandas as pd

def Estep(x: np.ndarray, G: int, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """EM algorithm expectation step. Here we estimate the latent variables based on the previous
    estimates of theta to build a responsibility matrix.

    Args:
        x (np.ndarray): Data matrix, (n, p)
        G (int): Number of classes
        pi (np.ndarray): Mixing weights, (G,)
        mu (np.ndarray): Mean values for each class, (p, G)
        sigma (np.ndarray): Shared covariance matrix, (p, p)
    
    Returns:
        np.ndarray: The responsibility matrix of shape (n, G)
    """
    resp = np.zeros((x.shape[0], G))
    for k in range(G):
        resp[:, k] = pi[k] * multivariate_normal_density(x, mu[:, k], sigma)
    return resp / resp.sum(axis=1).reshape(-1, 1)

def Mstep(x: np.ndarray, G: int, resp: np.ndarray, mu: np.ndarray):
    """EM algorithm maximization step.

    Args:
        x (np.ndarray): Data matrix, (n, p)
        G (int): Number of classes
        resp (np.ndarray): Responsibility matrix, (n, G)
        mu (np.ndarray): Mean values per dimension, (p, G)
    
    Returns:
        pi_new (np.ndarray): Updated mixing weights, (G,)
        mu_new (np.ndarray): Updated mean values per dimension, (p, G)
        sigma_new (np.ndarray): Updated covariance matrix, (p, p)
    """
    n = x.shape[0]
    # Pi
    pi_new = resp.sum(axis=0) / n
    # Mu
    mu_new = (x.T @ resp) / resp.sum(axis=0)
    # Sigma
    sigma_new = np.zeros(sigma.shape)
    for k in range(G):
        tmp = x.T - mu_new[:, k].reshape(-1, 1)
        #sigma_new += pi_new[k] * (resp[:, k] * A_mu) @ A_mu.T / resp[:, k].sum()
        sigma_new += pi_new[k] * tmp @ np.diag(resp[:, k]) @ tmp.T / resp[:, k].sum()
    return pi_new, mu_new, sigma_new

def loglik(x: np.ndarray, G: int, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """Calculate log likelihood, given distribution parameters.

    Args:
        x (np.ndarray): Input data, shape (n, p)
        G (int): Number of classes
        pi (np.ndarray): Mixing weights, shape (G,)
        mu (np.ndarray): Distribution means for each class, shape (p, G)
        sigma (np.ndarray): Shared covariance matrix, shape (p, p)

    Returns:
        float: log likelihood
    """
    ll = np.zeros(x.shape[0])
    for k in range(G):
        ll += pi[k] * multivariate_normal_density(x, mu[:, k], sigma)
    return np.log(ll).sum()

def multivariate_normal_density(x: np.ndarray, mu_k: np.ndarray, sigma: np.ndarray):
    """Evaluate multivariate normal probability density.
    This is used in the E-step and in the log-likelihood calculation.

    Args:
        x (np.ndarray): data, shape (n, p)
        mu_k (np.ndarray): mean for a given class, shape (p,)
        sigma (np.ndarray): covariance matrix, shape (p, p)

    Returns:
        np.ndarray: n-dimensional probability densities
    """
    A_mu = x.T - mu_k.reshape(-1, 1)
    exponent = - 0.5 * np.multiply(A_mu, np.linalg.inv(sigma) @ A_mu).sum(axis=0)
    return 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma))) * np.exp(exponent)

def myEM(data: np.ndarray, G: int, prob: np.ndarray,
         mean: np.ndarray, Sigma: np.ndarray, itmax: int):
    """Main EM algorithm

    Args:
        data (np.ndarray): Input data, shape (n, p)
        G (int): Number of classes
        prob (np.ndarray): Mixing weights, shape (G,)
        mean (np.ndarray): Distribution means for each class, shape (p, G)
        Sigma (np.ndarray): Shared covariance matrix, shape (p, p)
        itmax (int): Number of EM iterations to perform

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, float): probability vector, means, covariance and
                                                     log-likelihood
    """
    for _ in range(itmax):
        resp = Estep(data, G, prob, mean, Sigma)
        prob, mean, Sigma = Mstep(data, G, resp, mean)
        ll = loglik(data, G, prob, mean, Sigma)
    return prob, mean, Sigma, ll    

data = pd.read_csv('faithful.dat', header=0, sep='\s+')
data.head()
data = data.to_numpy()

# Case: G=2
G = 2
n = data.shape[0]
p1 = 10 / n
p2 = 1 - p1
mu1 = data[:10, :].mean(axis=0).reshape(-1, 1)
mu2 = data[10:, :].mean(axis=0).reshape(-1, 1)

sigma = 1 / n * (
           (data[:10].T - mu1) @ (data[:10].T - mu1).T + \
           (data[10:].T - mu2) @ (data[10:].T - mu2).T
        )

pi = np.array((p1, p2)) # Shape (G,)
mu = np.column_stack((mu1, mu2)) # Shape (p, G)

prob, mean, Sigma, ll = myEM(data, G, pi, mu, sigma, 20)
print("Case G=2")
print(f"prob\n{prob}\n\nmean\n{mean}\n\nSigma\n{Sigma}\n\nloglik\n{ll}\n")

# Case G=3
G = 3
p1 = 10 / n
p2 = 20 / n
p3 = 1 - p1 - p2
mu1 = data[:10, :].mean(axis=0).reshape(-1, 1)
mu2 = data[10:30, :].mean(axis=0).reshape(-1, 1)
mu3 = data[30:, :].mean(axis=0).reshape(-1, 1)
sigma = 1 / n * (
           (data[:10].T - mu1) @ (data[:10].T - mu1).T + \
           (data[10:30].T - mu2) @ (data[10:30].T - mu2).T + \
           (data[30:].T - mu3) @ (data[30:].T - mu3).T
        )

pi = np.array((p1, p2, p3)) # Shape (G,)
mu = np.column_stack((mu1, mu2, mu3)) # Shape (p, G)

prob, mean, Sigma, ll = myEM(data, G, pi, mu, sigma, 20)
print("Case G=3")
print(f"prob\n{prob}\n\nmean\n{mean}\n\nSigma\n{Sigma}\n\nloglik\n{ll}\n")