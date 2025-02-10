import numpy as np
from scipy.stats import norm
from math import sqrt, exp
from typing import Tuple, Dict

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calcule le prix d'une option européenne avec Black-Scholes.
    """
    if option_type not in ['call', 'put']:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calcule les grecques (Delta, Gamma, Vega, Theta, Rho) pour une option européenne avec Black-Scholes.
    """
    if option_type not in ['call', 'put']:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))) - (r * K * exp(-r * T) * norm.cdf(d2))
    rho = K * T * exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * exp(-r * T) * norm.cdf(-d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def binomial_tree(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call', steps: int = 100) -> float:
    """
    Calcule le prix d'une option américaine avec l'arbre binomial.
    """
    if option_type not in ['call', 'put']:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")

    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)

    # Calcul des prix du sous-jacent à l'échéance
    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]

    # Calcul des valeurs de l'option à l'échéance
    if option_type == 'call':
        payoff = [max(0, price - K) for price in prices]
    else:
        payoff = [max(0, K - price) for price in prices]

    # Rétropropagation dans l'arbre
    for _ in range(steps):
        payoff = [exp(-r * dt) * (p * payoff[j + 1] + (1 - p) * payoff[j]) for j in range(len(payoff) - 1)]
        # Exercice anticipé pour les options américaines
        if option_type == 'call':
            payoff = [max(payoff[j], prices[j] - K) for j in range(len(payoff))]
        elif option_type == 'put':
            payoff = [max(payoff[j], K - prices[j]) for j in range(len(payoff))]

    return payoff[0]

def calculate_strategy_payoff(S_range: np.ndarray, K1: float, K2: float, T: float, r: float, sigma: float, strategy: str) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Calcule le payoff et le coût d'une stratégie optionnelle.
    """
    if strategy == "Strangle":
        call_price = black_scholes(np.mean(S_range), K2, T, r, sigma, 'call')
        put_price = black_scholes(np.mean(S_range), K1, T, r, sigma, 'put')
        payoff = np.maximum(S_range - K2, 0) + np.maximum(K1 - S_range, 0) - (call_price + put_price)
        cost = call_price + put_price
        details = {"Call": call_price, "Put": put_price}
    elif strategy == "Bull Spread":
        call_price1 = black_scholes(np.mean(S_range), K1, T, r, sigma, 'call')
        call_price2 = black_scholes(np.mean(S_range), K2, T, r, sigma, 'call')
        payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0) - (call_price1 - call_price2)
        cost = call_price1 - call_price2
        details = {"Call 1 (Long)": call_price1, "Call 2 (Short)": call_price2}
    elif strategy == "Bear Spread":
        put_price1 = black_scholes(np.mean(S_range), K1, T, r, sigma, 'put')
        put_price2 = black_scholes(np.mean(S_range), K2, T, r, sigma, 'put')
        payoff = np.maximum(K1 - S_range, 0) - np.maximum(K2 - S_range, 0) - (put_price1 - put_price2)
        cost = put_price1 - put_price2
        details = {"Put 1 (Long)": put_price1, "Put 2 (Short)": put_price2}
    elif strategy == "Butterfly Spread":
        call_price1 = black_scholes(np.mean(S_range), K1, T, r, sigma, 'call')
        call_price2 = black_scholes(np.mean(S_range), K2, T, r, sigma, 'call')
        call_price3 = black_scholes(np.mean(S_range), (K1 + K2) / 2, T, r, sigma, 'call')
        payoff = np.maximum(S_range - K1, 0) - 2 * np.maximum(S_range - (K1 + K2) / 2, 0) + np.maximum(S_range - K2, 0) - (call_price1 - 2 * call_price3 + call_price2)
        cost = call_price1 - 2 * call_price3 + call_price2
        details = {"Call 1 (Long)": call_price1, "Call 2 (Short)": call_price2, "Call 3 (Long)": call_price3}
    elif strategy == "Collar":
        call_price = black_scholes(np.mean(S_range), K2, T, r, sigma, 'call')
        put_price = black_scholes(np.mean(S_range), K1, T, r, sigma, 'put')
        payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0) - (put_price - call_price)
        cost = put_price - call_price
        details = {"Call (Short)": call_price, "Put (Long)": put_price}
    else:
        raise ValueError("Stratégie non reconnue.")
    return payoff, cost, details
