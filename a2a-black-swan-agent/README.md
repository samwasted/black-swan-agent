# 🦢 Black Swan Agent (A2A)

The **Black Swan** agent is an institutional-grade, headless Agent-to-Agent (A2A) quantitative finance service. It acts as an adversarial stress-tester for algorithmic trading strategies, utilizing mathematical proofs and Walk-Forward Optimization (WFO) rather than simple heuristic guessing.

This agent is fully compliant with the Nasiko A2A network and executes rigorous Monte Carlo methods to evaluate true out-of-sample (OOS) robustness and tail risk.

## 🚀 Key Capabilities

1. **Dynamic Walk-Forward Optimization (WFO)**
   Automatically slices historical market data into sequential Train/Test blocks (e.g., 6 months train, 1 month test). It uses an asynchronous **Optuna** engine across `pandas-ta` to find optimal parameter bounds on the Train fold, enforcing them seamlessly on the Out-of-Sample Test fold without lookahead bias.
2. **Penalized Objective Function**
   Instead of purely maximizing returns, the WFO engine optimizes for maximum *risk-adjusted performance* by actively penalizing drawdowns: `Score = Sharpe - (MaxDD * Penalty_Factor)`.
3. **Adversarial Monte Carlo Lab**
   - **Connectivity MC**: Randomly drops successful/failed trades to simulate latency or broker slippage.
   - **Trade Shuffle MC**: Re-orders execution sequences to test if the strategy survived purely due to chronological luck.
   - **Day Sequence MC**: Shuffles entire market calendar days to destroy auto-correlation.
   - **Gaussian GBM MC**: Generates synthetic geometric Brownian motion curves based on the strategy's exact volatility standard deviation.
4. **Data Profiling & Context Narrative**
   Calculates advanced quant metrics (Hurst Exponent, Skewness, Kurtosis, VaR/CVaR) and pairs them with live market news headlines via `yfinance` to generate contextual, plain-text risk diagnostics.

## ⚙️ Installation & Usage

### 1. Local Python Setup

Requires Python 3.12+

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install "a2a-sdk[http-server]>=0.3.0" openai pydantic httpx click uvicorn
pip install numpy pandas pandas-ta optuna yfinance scipy

# Run the agent server
export OPENAI_API_KEY="sk-your-key-here"
python3 src/__main__.py --host localhost --port 5000
```

### 2. Docker Deployment

```bash
# Ensure OPENAI_API_KEY is placed in a local .env file
docker-compose up --build
```

## 📡 Evaluating a Strategy via API

Because Black Swan is a headless A2A worker, you prompt it via JSON-RPC. Send it dynamic range boundaries (e.g., `[5, 20]`) and it will automatically launch the Optuna WFO loop to find the best metrics inside those bounds.

```bash
curl -X POST http://localhost:5000/ \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-01",
      "timestamp": "2024-01-01T00:00:00Z",
      "role": "user",
      "parts": [{
        "text": "Run a quantitative robustness suite for an SMA strategy on BTC-USD. Use a 3y period. Optimize the fast_period between 5 and 20, and the slow_period between 21 and 100."
      }]
    },
    "metadata": {}
  }
}'
```

*(Note: Heavy parameter ranges spanning 2-3 years of Walk-Forward train/test folds calculate mathematically intensive Optuna trials and may take 15-30 seconds to return the final diagnostic text.)*

## 🛠 Supported `pandas-ta` Strategies
* **Moving Averages:** `sma`, `ema`, `wma`, `vwma`, `hma`, `rma`, `dema`, `tema`
* **Oscillators:** `rsi`, `cci`, `mfi`, `stoch`
* **MACD Family:** `macd`, `ppo`, `apo`
* **Bands:** `bbands`, `kc`, `dc`

## 💬 Response Formatting
The LLM has been heavily directed through `SYSTEM_PROMPT` engineering to output clean, structured PLAIN TEXT. It purposefully strips all Markdown formatting (`**`, `###`, etc.) to ensure seamless UI integration where Markdown renders might crash or look messy on terminal/dashboard outputs.
