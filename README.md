# dPolaris AI

A local AI trading intelligence system designed to help reach your $3M portfolio goal.

## Features

- **Claude AI Integration** - Direct Claude API for analysis + Claude CLI for web research
- **Local ML Models** - Train and use prediction models locally
- **Persistent Memory** - AI learns from your trading patterns over time
- **Broker Integration** - Real-time data from Webull, E*Trade, or Interactive Brokers
- **Background Daemon** - Continuous market scanning and alerts
- **Mac App Ready** - REST API + WebSocket for native app integration

## Quick Start

### 1. Install

```bash
cd dpolaris_ai
bash bootstrap_env.sh
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Initial Setup

```bash
.venv/bin/dpolaris setup
```

### 4. Start Using

```bash
# Interactive chat
.venv/bin/dpolaris chat

# Quick commands
.venv/bin/dpolaris scout                 # Scan for opportunities
.venv/bin/dpolaris analyze AAPL          # Deep analysis
.venv/bin/dpolaris quote NVDA            # Get quote
.venv/bin/dpolaris predict SPY           # ML prediction

# Start background daemon
.venv/bin/dpolaris start

# Start API server (for Mac app)
.venv/bin/dpolaris server
```

## Git + Dependencies

- Do not commit local virtual environments (`.venv`, `venv`); they are ignored by `.gitignore`.
- Keep dependencies in `requirements.txt` (and `pyproject.toml`).
- On a new machine or CI build, run:

```bash
bash bootstrap_env.sh
```

## Architecture

```
dpolaris_ai/
├── core/
│   ├── ai.py           # Main AI engine
│   ├── config.py       # Configuration management
│   ├── database.py     # SQLite persistence
│   ├── memory.py       # AI learning/memory system
│   └── claude_cli.py   # Claude CLI integration
├── ml/
│   ├── features.py     # Feature engineering
│   ├── trainer.py      # Model training
│   └── predictor.py    # Predictions
├── tools/
│   ├── market_data.py  # Yahoo Finance data
│   └── broker.py       # Broker integrations
├── api/
│   └── server.py       # FastAPI server
├── daemon/
│   └── scheduler.py    # Background jobs
└── cli/
    └── main.py         # CLI interface
```

## Commands

### Chat Commands

In chat mode, use these @ commands:

| Command | Description |
|---------|-------------|
| `@scout` | Scan for trading opportunities |
| `@analyze SYMBOL` | Deep analysis of a symbol |
| `@research SYMBOL` | Web research using Claude CLI |
| `@predict SYMBOL` | ML model prediction |
| `@train SYMBOL` | Train model for symbol |
| `@risk` | Portfolio risk assessment |
| `@regime` | Market regime analysis |
| `@performance` | Performance metrics |
| `@learn CONTENT` | Record a learning |

### CLI Commands

```bash
dpolaris start          # Start daemon
dpolaris stop           # Stop daemon
dpolaris status         # Check status
dpolaris chat           # Interactive chat
dpolaris scout          # Quick opportunity scan
dpolaris analyze AAPL   # Analyze symbol
dpolaris research NVDA  # Web research
dpolaris predict SPY    # ML prediction
dpolaris train QQQ      # Train model
dpolaris quote TSLA     # Get quote
dpolaris watchlist      # View watchlist
dpolaris watch-add MSFT # Add to watchlist
dpolaris journal        # View trade journal
dpolaris risk           # Risk assessment
dpolaris performance    # Performance stats
dpolaris backup         # Create backup
dpolaris server         # Start API server
```

## Configuration

Edit `~/dpolaris_data/config/settings.yaml`:

```yaml
goal:
  target: 3000000
  starting_capital: 100000

risk:
  max_position_size_percent: 5.0
  max_drawdown_percent: 15.0

watchlist:
  - SPY
  - QQQ
  - AAPL
  - NVDA
```

## Broker Setup

### Webull (Real-time Data)

```bash
pip install webull
```

### E*Trade

1. Apply for developer access at https://developer.etrade.com
2. Get consumer key and secret
3. Configure in settings.yaml

### Interactive Brokers

```bash
pip install ib_insync
```

Then run TWS or IB Gateway with API enabled.

## API Endpoints

When running `dpolaris server`:

- `GET /api/portfolio` - Portfolio summary
- `GET /api/positions` - Open positions
- `GET /api/journal` - Trade journal
- `GET /api/watchlist` - Watchlist
- `POST /api/chat` - Chat with AI
- `GET /api/scout` - Opportunity scan
- `GET /api/market/quote/{symbol}` - Get quote
- `WS /ws/portfolio` - Real-time portfolio updates
- `WS /ws/chat` - Streaming chat

## Data Portability

All data is stored in `~/dpolaris_data/`. To migrate to a new machine:

1. Copy the entire folder
2. Install dPolaris
3. Run `dpolaris status`

Your AI memories, trade history, and models will be preserved.

## License

MIT
