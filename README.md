# 🦢 Black Swan Agent
### Adversarial Robustness Testing for Algorithmic Trading Systems

**Black Swan** is a headless, institutional-grade adversarial testing agent for algorithmic trading strategies.

It does not predict markets.  
It **attempts to break your strategy**.

By combining:
- Walk-Forward Optimization (WFO)
- Monte Carlo adversarial simulations
- Strict anti-lookahead backtesting
- High-performance Numba-accelerated JIT kernels

Black Swan produces **mathematically defensible robustness reports**, not heuristic opinions.

> If your strategy survives Black Swan, it’s worth considering. If it doesn’t, it never was.

---

Black Swan is not a backtesting tool.  
It is a **strategy adversary**.

---

## 🎯 Why Black Swan Exists

Most retail and even semi-professional backtesting systems suffer from:
- **Lookahead bias**
- **Overfitting** via static optimization over the entire dataset
- **Ignoring execution uncertainty** (assuming perfect fills)
- **Ignoring tax drag** (which can destroy many "profitable" strategies)

Black Swan addresses these by:
- Enforcing strict `t+1` execution semantics
- Using continuous Walk-Forward Optimization instead of global curve-fitting
- Stress-testing strategies across multiple adversarial Monte Carlo regimes
- **Accurate execution modeling:** Native support for per-trade fees and slippage.
- **Tax-Aware Outcomes:** Real-world capital gains tracking (STCG/LTCG) for US, UK, and India.

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

## 👥 Intended Users

- **Quant Developers** validating strategy robustness and out-of-sample stability.
- **Researchers** studying overfitting, parameter sensitivity, and regime transitions.
- **Systems Engineers** building automated trading frameworks requiring high-integrity validation.

*Not designed for: Retail signal generation or manual discretionary trading decisions.*

---

## 🏗 Architecture Overview

```text
User JSON-RPC Request
          ↓
A2A HTTP Server (__main__.py)
          ↓
OpenAI Agent Executor (with session memory)
          ↓
LLM Confirmation Flow (Fees/Slippage/Tax)
          ↓
QuantToolset (Numba JIT Engine)
          ↓
[WFO + Monte Carlo + Tax Engine]
          ↓
LLM Summary Formulation -> User
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/samwasted/nasiko-ai-agent
cd nasiko-ai-agent/a2a-black-swan-agent

# 2. Setup environment
python -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Set credentials
export OPENAI_API_KEY="your_openai_api_key"

# 5. Run the agent
python -m src
```

---

## 🔬 Quantitative Engine

### Walk-Forward Optimization (WFO)
Rather than optimizing parameters over the entire dataset, Black Swan perpetually trains and validates parameters forward through time via Optuna.

- **Train Window:** 6 months  
- **Test Window:** 1 month (strictly out-of-sample)  
- **Optimizer:** Optuna (30 trials per train window)  
- **Objective Function:** `Sharpe - (2.0 × Max Drawdown)`
- **Engine:** Numba-accelerated `@njit` kernels for near-C execution speeds.
- **Costs:** Variable fees and slippage applied at each position change.

### Signal Engine (Dynamic)
Supports dynamic parameterization across standard implementations:
- **Moving Averages:** `sma`, `ema`, `wma`, `hma`, etc.
- **Oscillators:** `rsi`, `cci`, `mfi`, `stoch`
- **MACD Family:** `macd`, `ppo`, `apo`
- **Bands:** `bbands`, `kc`, `dc`

*Note: All signals strictly enforce **t+1 execution lag** to eliminate lookahead bias.*

### Tax Engine (Holding-Period Aware)
Black Swan tracks every discrete trade to determine holding duration. It applies country-specific tax regimes:
- **India:** STCG (20%) / LTCG (12.5%) after 365 days.
- **USA:** STCG (37% max) / LTCG (20%) after 365 days.
- **UK:** 20% Capital Gains.

---

## 🎲 Adversarial Monte Carlo Lab

Black Swan applies multiple stress regimes to the Out-of-Sample equity curve:

1. **Execution Failure Simulation (Connectivity)**
   - Randomly drops 20% of trades to simulate latency, broker disconnects, or extreme slippage.
2. **Trade Order Randomization**
   - Tests if the strategy's survival was dependent on a specific historical chronological sequence.
3. **Temporal Disorder Simulation**
   - Shuffles market days individually to destroy long-term autocorrelation and trend dependency.
4. **Synthetic Market Generation (GBM)**
   - Generates large-scale Gaussian GBM price paths calibrated to observed volatility.

> **Goal:** Detect fragility, not optimize returns.

---

## ⚡ Performance Characteristics

| Factor | Impact |
|--------|--------|
| **WFO + Optuna** | High CPU thread saturation during Train blocks. |
| **Monte Carlo Labs** | High execution latency (15–30 seconds to return response). |
| **yFinance API** | Dependent on unofficial endpoints, subject to rate-limiting. |
| **Numba JIT**    | Very fast simulation loop, but incurs a ~3-second compilation hit on first startup. |

### Modeling Assumptions

To isolate structural robustness, the current engine:
- **Full Capital Allocation:** Assumes unweighted 100% allocations (no position sizing) to expose pure strategy volatility.

---

## 🚀 Example Usage (A2A Confirmation Flow)

Because Black Swan is a headless A2A logic engine with built-in financial safeguards, it uses a mandatory two-step confirmation process. 

**1. Initial Request:**
Pass dynamic array boundaries and explicitly request custom fees, slippage, and tax regimes.
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
      "timestamp": "2026-03-29T00:00:00Z",
      "role": "user",
      "parts": [{
        "text": "Run a robustness suite for an SMA strategy on BTC-USD. Use a 2y period. Optimize the fast_period between 5 and 20, and the slow_period between 21 and 100. Assume 0.1% fees, 0.05% slippage, and apply the US tax regime."
      }]
    }
  }
}'
```

The agent will respond with a breakdown of execution parameters and ask "Proceed? (yes/no)". You will also receive a `contextId` in the JSON response.

**2. Confirmation (Reply 'yes'):**
Take the `contextId` from the previous response and reply "yes" to push the strategy into the Numba WFO engine.
```bash
curl -X POST http://localhost:5000/ \
-H "Content-Type: application/json" \
-d '{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-02",
      "contextId": "YOUR_CONTEXT_ID_HERE",
      "timestamp": "2026-03-29T00:01:00Z",
      "role": "user",
      "parts": [{
        "text": "yes"
      }]
    }
  }
}'
```

### Sample Response Snippet (Data Payload)

```json
{
  "ticker": "BTC-USD",
  "oos_metrics": {
    "cagr": 0.184,
    "max_drawdown": 0.271,
    "sharpe": 1.12
  },
  "monte_carlo": {
    "connectivity_avg": 0.112,
    "worst_case_var95": -0.354,
    "gbm_fail_rate": "12.5%"
  },
  "analytical_narrative": "Strategy displays strong autocorrelation resistance..."
}
```

---

## 🔮 Future Work

- [ ] Position sizing (Kelly Criterion, ATR-based risk targeting).
- [ ] Multi-asset portfolio simulation and co-integration testing.
- [ ] Market regime detection (bull/bear/sideways segregation).
- [ ] GPU-accelerated Monte Carlo execution paths.
