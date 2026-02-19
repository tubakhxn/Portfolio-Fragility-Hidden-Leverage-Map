# 3D Portfolio Fragility & Hidden Leverage Map

## What is this project about?

This project is an advanced risk analytics dashboard for multi-asset portfolios. It enables users to:
- Construct portfolios with dynamic leverage, volatility targeting, and optional convex payoffs
- Simulate realistic stress scenarios: volatility shocks, correlation spikes, and liquidity haircuts
- Visualize portfolio fragility and hidden leverage in 3D (vol shock × corr shock → portfolio loss)
- Compute and display key risk metrics: drawdowns, convexity, fragility index, tail loss, and hidden leverage
- Interactively explore risk using a modern Streamlit web UI

The system is built with Python, NumPy, Pandas, Plotly, and Streamlit, and implements real nonlinear risk logic (no simplifications).

## How to fork this project

1. Click the **Fork** button at the top right of this repository on GitHub.
2. Clone your forked repository to your local machine:
   ```
   git clone https://github.com/YOUR-USERNAME/Portfolio-Fragility-Hidden-Leverage-Map.git
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Creator/Dev

### tubakhxn
