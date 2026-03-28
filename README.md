# 🦢 Black Swan Agent

**Black Swan** is a headless, institutional-grade adversarial testing agent for algorithmic trading strategies.

It does not predict markets.  
It **attempts to break your strategy**.

By combining:
- Walk-Forward Optimization (WFO)
- Monte Carlo adversarial simulations
- Strict anti-lookahead backtesting

Black Swan produces **mathematically defensible robustness reports**, not heuristic opinions.

> If your strategy survives Black Swan, it’s worth considering. If it doesn’t, it never was.

---

## 🎯 Why Black Swan Exists

Most retail and even semi-professional backtesting systems suffer from:
- **Lookahead bias**
- **Overfitting** via static optimization over the entire dataset
- **Ignoring execution uncertainty** (assuming perfect fills)
- **Misleading single-path equity curves**

Black Swan addresses these by:
- Enforcing strict `t+1` execution semantics
- Using continuous Walk-Forward Optimization instead of global curve-fitting
- Stress-testing strategies across multiple adversarial Monte Carlo regimes
- Evaluating true statistical robustness, not peak performance

---

## 🧠 Design Philosophy

Black Swan follows one rule:

> **"Assume your strategy is wrong. Prove otherwise."**

It does not:
- Predict prices
- Guarantee future returns
- Optimize explicitly for best-case outcomes

It does:
- Stress-test underlying assumptions
- Penalize fragility and drawdowns
- Reward empirical robustness

---

## 🏗 Architecture Overview

```text
User JSON-RPC Request
          ↓
A2A HTTP Server (__main__.py)
          ↓
OpenAI Agent Executor
          ↓
LLM (Intent & Decision Layer)
          ↓
QuantToolset (Deterministic Engine)
          ↓
[WFO + Monte Carlo + Metrics]
          ↓
LLM Summary Formulation -> User
```

## 📂 Code Structure

| File | Responsibility |
|------|----------------|
| `__main__.py` | Bootstraps the A2A server via FastAPI & Uvicorn. Registers skills. |
| `openai_agent_executor.py` | Core wrapper handling JSON-RPC schemas and physical tool execution binding. |
| `openai_agent.py` | Defines the LLM instruction boundaries, system prompts, and tool lists. |
| `quant_toolset.py` | The deterministic heavy mathematical engine (No LLM dependencies). |

---

## 🔬 Quantitative Engine

### Walk-Forward Optimization (WFO)
Rather than optimizing parameters over the entire dataset, Black Swan perpetually trains and validates parameters forward through time.

- **Train Window:** 6 months  
- **Test Window:** 1 month (strictly out-of-sample)  
- **Optimizer:** Optuna (30 trials per train window)  
- **Objective Function:** `Sharpe - (2.0 × Max Drawdown)`

### Signal Engine (Dynamic)
Supports dynamic parameterization across standard implementations:
- **Moving Averages:** `sma`, `ema`, `wma`, `hma`, etc.
- **Oscillators:** `rsi`, `cci`, `mfi`, `stoch`
- **MACD Family:** `macd`, `ppo`, `apo`
- **Bands:** `bbands`, `kc`, `dc`

*Note: All signals strictly enforce **t+1 execution lag** to eliminate lookahead bias.*

---

## 🎲 Adversarial Monte Carlo Lab

Black Swan applies multiple stress regimes to the Out-of-Sample equity curve:

1. **Execution Failure Simulation (Connectivity)**
   - Randomly drops 20% of trades to simulate latency, broker disconnects, or slippage.
2. **Trade Order Randomization**
   - Tests if the strategy survived purely due to chronological sequence luck.
3. **Temporal Disorder Simulation**
   - Shuffles market days individually to destroy long-term autocorrelation.
4. **Synthetic Market Generation (GBM)**
   - Casts millions of Gaussian GBM price paths using observed volatility.

> **Goal:** Detect fragility, not optimize returns.

---

## ⚡ Performance Characteristics

| Factor | Impact |
|--------|--------|
| **WFO + Optuna** | High CPU thread saturation during Train blocks. |
| **Monte Carlo Labs** | High execution latency (15–30 seconds to return response). |
| **yFinance API** | Dependent on unofficial endpoints, subject to rate-limiting. |

### Known Constraints
- No spread/cost modeling (yet). Sizing assumes unweighted 100% allocations.
- Refuses to optimize on datasets containing fewer than 100 periods.
- Not suitable for ultra-high-frequency (HFT) modeling.

---

## 🚀 Example Usage

Because Black Swan is a headless A2A logic engine, you call it programmatically. Pass dynamic array boundaries directly in the prompt to trigger the optimization engine:

**Request:**
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
        "text": "Run a robustness suite for an SMA strategy on BTC-USD. Use a 3y period. Optimize the fast_period between 5 and 20, and the slow_period between 21 and 100."
      }]
    },
    "metadata": {}
  }
}'
```

**Output Architecture:**
1. Evaluated optimal parameter ranges per historical epoch.
2. Out-of-sample (OOS) equity curve metrics.
3. Monte Carlo robustness distribution models.
4. An LLM-synthesized plain-text narrative summarizing risk, context, and news.

---

## 🔮 Future Work

- [ ] Slippage and variable commission cost modeling.
- [ ] Position sizing (Kelly Criterion, ATR-based risk targeting).
- [ ] Multi-asset portfolio simulation.
- [ ] Market regime detection constraints (bull/bear/sideways segregation).
- [ ] GPU-accelerated Monte Carlo execution.
