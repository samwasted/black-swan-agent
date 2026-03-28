# 🦢 Black Swan Agent – Comprehensive Architecture Audit

The **Black Swan** agent is an institutional-grade, headless, Agent-to-Agent (A2A) quantitative finance service. It is designed to act as an adversarial stress-tester for algorithmic trading strategies by using mathematical proofs instead of heuristic guessing.

Here is the exact file-by-file breakdown, execution flow, practices, and caveats currently powering it.

---

## 📂 1. Directory Structure & Files

### `src/__main__.py`
* **Purpose:** The main entry point for the Application.
* **How it works:** It uses `a2a-sdk` and `uvicorn` (FastAPI) to spin up an HTTP server listening on port `5000`. It registers the agent (named `"black-swan"`) using the `OpenAIAgentExecutor` wrapper class.
* **Caveat:** It requires the `.env` file to be populated with `OPENAI_API_KEY` for the executor to authenticate.

### `src/openai_agent_executor.py`
* **Purpose:** The bridge wrapper between the A2A HTTP Server Schema and the official `openai` Python SDK.
* **How it works:** It receives a `message/send` JSON-RPC webhook, processes the user’s text via the OpenAI API (typically using `gpt-4o-mini` or similar depending on configuration), parses any tool calls the LLM decides to throw (like `run_robustness_suite`), physically executes those Python methods, and hands the tools' JSON output back to the LLM to write the final summary.
* **Practice:** It uses robust JSON parsing to ensure stringified inputs are perfectly coerced into Python `dicts` before handing them to `QuantToolset`.

### `src/openai_agent.py`
* **Purpose:** Houses the configuration and "Brain" parameters (System Prompt) of the LLM.
* **How it works:** It exports `create_agent()`, linking the physical `QuantToolset()` tools and defining the massive `SYSTEM_PROMPT`.
* **Important Practice:** This prompt *strictly enforces* that the agent does not hallucinate financial numbers. It acts purely as a "data-routing service", blindly funneling inputs into the `QuantToolset` tool, reading the deterministic math JSON output, and summarizing it alongside recent `yfinance` news headlines as an analytical brief.

### `src/quant_toolset.py` (The Heavy Lifter)
* **Purpose:** Pure Python mathematical analysis. No LLMs live here. 
* **Key Components:**
  * **`run_robustness_suite`**: The solitary async public method exposed to the LLM. It spins off the giant workload into a ThreadPool (`run_in_executor`) to prevent blocking the async web server.
  * **`_fetch_data`**: Pulls raw OHLCV off Yahoo Finance (`yfinance`) and forward-fills (`ffill`) missing data gaps to prevent NaN crashes.
  * **`_run_wfo`**: The Walk-Forward Optimization Engine. This slices data into 6-month train blocks. Inside each train block, it spins up an async `optuna` study running 30 parameter combinations against `pandas-ta` indicators to maximize the `Sharpe - (MaxDD * 2.0)` score. It then takes those optimal bounds and trades the next immediately adjacent 1-month test block to form *True Out-of-Sample* (OOS) data.
  * **`_generate_signals_dynamic`**: Evaluates parameters over `pandas-ta` natively. It translates arrays like `[5, 20]` dynamically across `sma`, `rsi`, `macd`, and `bbands` families.
  * **`_compute_baseline_metrics`**: Standard quant metrics (CAGR, Max Drawdown, Sharpe, Sortino).
  * **Adversarial Lab**: 
    1. *Connectivity MC*: Drops 20% of trades randomly to simulate bad execution/slippage.
    2. *Trade Shuffle MC*: Re-orders trades to see if the strategy was just "lucky" with sequence order.
    3. *Day Shuffle MC*: Re-orders all calendar days (stressing time-based autocorrelation).
    4. *Gaussian GBM MC*: Fully synthetic Brownian Motion Monte Carlo based on the volatility standard deviation profile.
* **Practice / Caveat:** Notice that inside `_run_backtest`, the code heavily enforces `t+1` shift (`positions[1:] = sig_arr[:-1]`). This is the industry-standard way of preventing **lookahead bias**. A signal generated on Monday's close is executed on Tuesday's open/close return.

---

## ⚡ Execution Flow
1. **User Request**: `"Optimize an SMA on AAPL from 10 to 50"` -> `POST /`
2. **A2A Server (`__main__`)**: Translates payload -> hands to `executor`.
3. **OpenAI (`openai_agent`)**: GPT-4 reads system prompt, decides it needs empirical data, triggers `run_robustness_suite(ticker="AAPL", strategy_params='{"fast_period":[10, 50]}')`.
4. **QuantToolset**:
   - Downloads AAPL from Yahoo Finance.
   - Slices into 6m Train / 1m Test blocks.
   - Optuna tests millions of variations and saves the best ones.
   - Stitches together the 1-month Out-of-Sample equity curve.
   - Runs 2000x Monte Carlos on that OOS curve.
   - Pulls 5 Yahoo Finance News headlines.
   - Spits out a massive JSON dictionary.
5. **OpenAI (`executor`)**: LLM reads the JSON dictionary and the Yahoo news headlines. Synthesizes a risk-aware narrative. Returns final text to User.

---

## 🚧 Known Caveats & Pitfalls

1. **Computational Heaviness**: Because the agent now runs dynamic Optuna Walk-Forward Optimizations over 2,000 Monte Carlo loops per permutation, requests can take **15–30 seconds**. This will trigger timeout errors if the client requesting the API has a strict 10s wait period. 
2. **Dependent on Yahoo Finance API Limits**: `yfinance` is legally an unofficial scraper. If the server executes hundreds of tests per hour, Yahoo will temporary rate-limit/ban the server IP.
3. **Slippage Ignored**: The current WFO engine simulates perfect `t+1` market execution with `0.00` slippage and fees. The "Connectivity Drop MC" attempts to penalize this, but it is not a true spread simulation.
4. **Data Size**: The WFO Engine refuses to process datasets smaller than 100 days (`Got X rows; need at least 100 for WFO.`). A 2-year or 3-year `period` parameter flag is highly recommended for best results.
