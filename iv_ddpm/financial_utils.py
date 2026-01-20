import numpy as np
from scipy.stats import norm

# --- Financial Utilities ---

def calculate_penalties_vectorized(iv_tensor, m_grid, ttm_grid, P_T, P_K, PB_K, r_val=0.0):
    if iv_tensor.ndim == 4:
        iv_tensor = np.squeeze(iv_tensor, axis=1)

    M, T = np.meshgrid(m_grid, ttm_grid, indexing='ij')
    M = M[np.newaxis, :, :]
    T = T[np.newaxis, :, :]
    price_tensor = smallBS(M, T, iv_tensor, r_val) # Shape: (N, 9, 9)

    P1 = np.maximum(0, price_tensor @ P_T)
    P2 = np.maximum(0, np.einsum('ij,njk->nik', P_K, price_tensor))
    P3 = np.maximum(0, np.einsum('ij,njk->nik', PB_K, price_tensor))

    total_penalty_per_surface = np.sum(P1 + P2 + P3, axis=(1, 2))

    return P1, P2, P3, total_penalty_per_surface

def smallBS(m,tau,sigma,r):

    d1 = (-np.log(m)+tau*(r+0.5*sigma*sigma))/(sigma*np.sqrt(tau))
    d2 = d1-sigma*np.sqrt(tau)
    price = norm.cdf(d1)-m*norm.cdf(d2)*np.exp(-r*tau)
    return price

def penalty_mutau(mu,T):

    P_T = np.zeros((len(T),len(T)))
    P_K = np.zeros((len(mu),len(mu)))
    PB_K = np.zeros((len(mu),len(mu)))
    #P_T first, the last one is zero
    for j in np.arange(0,len(T)-1,1):
        P_T[j,j] = T[j]/(T[j+1]-T[j])
        P_T[j+1,j] = -T[j]/(T[j+1]-T[j])
    #now P_K and then PB_K
    for i in np.arange(0,len(mu)-1,1):
        P_K[i,i] = -1/(mu[i+1]-mu[i])
        P_K[i,i+1] = 1/(mu[i+1]-mu[i])
    #PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    #once we fix the grid we have to run this function only once so it doesn't matter much
    for i in np.arange(1,len(mu)-1,1):
        PB_K[i,i-1] = -(mu[i+1]-mu[i]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i,i] = (mu[i+1] - mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i,i+1] = -(mu[i]-mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
    return P_T,P_K,PB_K
