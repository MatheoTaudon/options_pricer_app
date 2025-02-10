import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import black_scholes, black_scholes_greeks, binomial_tree, calculate_strategy_payoff

# Configuration de la page
st.set_page_config(page_title="Pricer d'Options", layout="wide")
st.title("Pricer d'Options et Stratégies Optionnelles")

# Sidebar pour les paramètres généraux
st.sidebar.header("Paramètres Généraux")
S = st.sidebar.number_input("Prix du Sous-Jacent (S)", value=100.0)
T = st.sidebar.number_input("Maturité (T en années)", value=1.0)
r = st.sidebar.number_input("Taux d'Intérêt (r)", value=0.05)
sigma = st.sidebar.number_input("Volatilité (σ)", value=0.2)

# Choix du type d'option ou de stratégie
st.header("Type d'Option ou Stratégie")
option_type = st.selectbox("Choisir le type", ["Option Vanille", "Stratégie Optionnelle"])

if option_type == "Option Vanille":
    st.subheader("Paramètres de l'Option Vanille")
    K = st.number_input("Strike (K)", value=100.0)
    option_style = st.selectbox("Style de l'Option", ["Européenne", "Américaine"])
    option_direction = st.selectbox("Type d'Option", ["Call", "Put"])
    pricing_method = st.selectbox("Méthode de Pricing", ["Black-Scholes", "Arbre Binomial"])

    if st.button("Calculer le Prix"):
        if pricing_method == "Black-Scholes":
            price = black_scholes(S, K, T, r, sigma, option_direction.lower())
            greeks = black_scholes_greeks(S, K, T, r, sigma, option_direction.lower())
        elif pricing_method == "Arbre Binomial":
            price = binomial_tree(S, K, T, r, sigma, option_direction.lower())
            greeks = {}  # Les grecques ne sont pas calculés pour l'arbre binomial ici
        st.write(f"Prix de l'option : {price:.2f}")
        if greeks:
            st.write("Grecques :")
            st.write(f"Delta : {greeks['delta']:.4f}")
            st.write(f"Gamma : {greeks['gamma']:.4f}")
            st.write(f"Vega : {greeks['vega']:.4f}")
            st.write(f"Theta : {greeks['theta']:.4f}")
            st.write(f"Rho : {greeks['rho']:.4f}")

        # Affichage du payoff
        S_range = np.linspace(50, 150, 100)
        if option_direction == "Call":
            payoff = np.maximum(S_range - K, 0) - price
        else:
            payoff = np.maximum(K - S_range, 0) - price
        fig, ax = plt.subplots(figsize=(8, 4))  # Taille réduite du graphique
        ax.plot(S_range, payoff, label=f"Payoff {option_direction}")
        ax.set_xlabel("Prix du Sous-Jacent")
        ax.set_ylabel("Payoff")
        ax.set_title(f"Payoff de l'Option {option_direction} (Net)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

elif option_type == "Stratégie Optionnelle":
    st.subheader("Paramètres de la Stratégie")
    strategy = st.selectbox("Choisir une Stratégie", ["Strangle", "Bull Spread", "Bear Spread", "Butterfly Spread", "Collar"])
    
    if strategy in ["Strangle", "Collar"]:
        K1 = st.number_input("Strike 1 (K1) - Put", value=90.0)
        K2 = st.number_input("Strike 2 (K2) - Call", value=110.0)
    elif strategy in ["Bull Spread", "Bear Spread"]:
        K1 = st.number_input("Strike 1 (K1) - Long", value=90.0)
        K2 = st.number_input("Strike 2 (K2) - Short", value=110.0)
    elif strategy == "Butterfly Spread":
        K1 = st.number_input("Strike 1 (K1) - Long Call", value=90.0)
        K2 = st.number_input("Strike 2 (K2) - Short Call", value=100.0)
        K3 = st.number_input("Strike 3 (K3) - Long Call", value=110.0)

    if st.button("Calculer le Prix"):
        S_range = np.linspace(50, 150, 100)
        payoff, cost, details = calculate_strategy_payoff(S_range, K1, K2, T, r, sigma, strategy)
        st.write(f"Coût de la stratégie : {cost:.2f}")
        st.write("Détails des primes :")
        for key, value in details.items():
            st.write(f"{key} : {value:.2f}")
        fig, ax = plt.subplots(figsize=(8, 4))  # Taille réduite du graphique
        ax.plot(S_range, payoff, label=strategy)
        ax.set_xlabel("Prix du Sous-Jacent")
        ax.set_ylabel("Payoff")
        ax.set_title(f"Payoff de la Stratégie {strategy} (Net)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
