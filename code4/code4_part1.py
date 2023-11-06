import numpy as np
import pandas as pd

def Estep(x: pd.DataFrame, G: int, pi: np.ndarray, mu: pd.DataFrame, sigma: pd.DataFrame):
    """EM algorithm expectation step. Here we estimate the latent variables based on the previous
    estimates of theta to build a responsibility matrix.

    Args:
        x (pd.DataFrame): Data matrix, (n, p)
        G (int): Number of classes
        pi (np.ndarray): Mixing weights, (G,)
        mu (pd.DataFrame): Mean values for each class, (p, G)
        sigma (pd.DataFrame): Shared covariance matrix, (p, p)
    
    Returns:
        np.ndarray: The responsibility matrix of shape (n, G)
    """
    resp = np.zeros((x.shape[0], G))
    for k in range(G):
        resp[:, k] = pi[k] * multivariate_normal_density(x, mu.iloc[:, k], sigma)
    return resp / resp.sum(axis=1).reshape(-1, 1)

def Mstep(x: pd.DataFrame, G: int, resp: np.ndarray, sigma: pd.DataFrame):
    """EM algorithm maximization step.

    Args:
        x (np.ndarray): Data matrix, (n, p)
        G (int): Number of classes
        resp (np.ndarray): Responsibility matrix, (n, G)
        sigma (pd.DataFrame): Shared covariance matrix, (p, p)
    
    Returns:
        pi_new (np.ndarray): Updated mixing weights, (G,)
        mu_new (pd.DataFrame): Updated mean values per dimension, (p, G)
        sigma_new (pd.DataFrame): Updated covariance matrix, (p, p)
    """
    n = x.shape[0]
    # Pi
    pi_new = resp.sum(axis=0) / n
    # Mu
    mu_new = (x.T @ resp) / resp.sum(axis=0)
    # Sigma
    sigma_new = sigma.copy()
    sigma_new.values[:, :] = 0
    for k in range(G):
        tmp = x.T - mu_new.values[:, k].reshape(-1, 1)
        tmp = tmp.to_numpy()
        sigma_new += pi_new[k] * tmp @ np.diag(resp[:, k]) @ tmp.T / resp[:, k].sum()
    return pi_new, mu_new, sigma_new

def loglik(x: pd.DataFrame, G: int, pi: np.ndarray, mu: pd.DataFrame, sigma: pd.DataFrame):
    """Calculate log likelihood, given distribution parameters.

    Args:
        x (pd.DataFrame): Input data, shape (n, p)
        G (int): Number of classes
        pi (np.ndarray): Mixing weights, shape (G,)
        mu (pd.DataFrame): Distribution means for each class, shape (p, G)
        sigma (pd.DataFrame): Shared covariance matrix, shape (p, p)

    Returns:
        float: log likelihood
    """
    ll = np.zeros(x.shape[0])
    for k in range(G):
        ll += pi[k] * multivariate_normal_density(x, mu.iloc[:, k], sigma)
    return np.log(ll).sum()

def multivariate_normal_density(x: pd.DataFrame, mu_k: pd.DataFrame, sigma: pd.DataFrame):
    """Evaluate multivariate normal probability density.
    This is used in the E-step and in the log-likelihood calculation.

    Args:
        x (pd.DataFrame): data, shape (n, p)
        mu_k (pd.DataFrame): mean for a given class, shape (p,)
        sigma (pd.DataFrame): covariance matrix, shape (p, p)

    Returns:
        np.ndarray: n-dimensional probability densities
    """
    A_mu = (x.T - mu_k.values.reshape(-1, 1)).to_numpy()
    exponent = - 0.5 * np.multiply(A_mu, np.linalg.inv(sigma) @ A_mu).sum(axis=0)
    return 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma))) * np.exp(exponent)

def myEM(data: pd.DataFrame, G: int, prob: np.ndarray,
         mean:pd.DataFrame, Sigma: pd.DataFrame, itmax: int):
    """Main EM algorithm

    Args:
        data (pd.DataFrame: Input data, shape (n, p)
        G (int): Number of classes
        prob (np.ndarray): Mixing weights, shape (G,)
        mean (pd.DataFrame): Distribution means for each class, shape (p, G)
        Sigma (pd.DataFrame): Shared covariance matrix, shape (p, p)
        itmax (int): Number of EM iterations to perform

    Returns:
        (np.ndarray, pd.DataFrame, pd.DataFrame, float): probability vector, means, covariance and
                                                         log-likelihood
    """
    for _ in range(itmax):
        resp = Estep(data, G, prob, mean, Sigma)
        prob, mean, Sigma = Mstep(data, G, resp, Sigma)
        ll = loglik(data, G, prob, mean, Sigma)
    return prob, mean, Sigma, ll    

data = pd.read_csv('faithful.dat', header=0, sep='\s+')
data.head()

# Case: G=2
G = 2
n = data.shape[0]
p1 = 10 / n
p2 = 1 - p1
mu1 = data.iloc[:10, :].mean(axis=0)
mu2 = data.iloc[10:, :].mean(axis=0)

sigma = 1 / n * (
           (data.iloc[:10, :].T - mu1.to_numpy().reshape(-1, 1)) @
           (data[:10].T - mu1.to_numpy().reshape(-1, 1)).T + \
           (data.iloc[10:, :].T - mu2.to_numpy().reshape(-1, 1)) @
           (data[10:].T - mu2.to_numpy().reshape(-1, 1)).T
        )

pi = np.array((p1, p2)) # Shape (G,)
mu = pd.DataFrame({"0": mu1, "1": mu2}) # Shape (p, G)

prob, mean, Sigma, ll = myEM(data, G, pi, mu, sigma, 20)
print("Case G=2")
print(f"prob\n{prob}\n\nmean\n{mean}\n\nSigma\n{Sigma}\n\nloglik\n{ll:.3f}\n")

# Case G=3
G = 3
p1 = 10 / n
p2 = 20 / n
p3 = 1 - p1 - p2
mu1 = data.iloc[:10, :].mean(axis=0)
mu2 = data.iloc[10:30, :].mean(axis=0)
mu3 = data.iloc[30:, :].mean(axis=0)
sigma = 1 / n * (
           (data.iloc[:10].T - mu1.to_numpy().reshape(-1, 1)) @
           (data.iloc[:10].T - mu1.to_numpy().reshape(-1, 1)).T + \
           (data.iloc[10:30].T - mu2.to_numpy().reshape(-1, 1)) @
           (data.iloc[10:30].T - mu2.to_numpy().reshape(-1, 1)).T + \
           (data.iloc[30:].T - mu3.to_numpy().reshape(-1, 1)) @
           (data.iloc[30:].T - mu3.to_numpy().reshape(-1, 1)).T
        )

pi = np.array((p1, p2, p3)) # Shape (G,)
mu = pd.DataFrame({"0": mu1, "1": mu2, "2": mu3}) # Shape (p, G)

prob, mean, Sigma, ll = myEM(data, G, pi, mu, sigma, 20)
print("Case G=3")
print(f"prob\n{prob}\n\nmean\n{mean}\n\nSigma\n{Sigma}\n\nloglik\n{ll:.3f}\n")