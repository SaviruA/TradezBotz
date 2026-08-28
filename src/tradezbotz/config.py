"""Configuration loading and the live-trading safety gate.

The gate is deliberately annoying. Reaching live mode requires three
independent, explicit signals; any one of them missing keeps the bot on
paper. See `LIVE_ACK` for the exact value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"

#: The literal string that must be in TRADEZBOTZ_ALLOW_LIVE to arm live trading.
LIVE_ACK = "i-understand-this-trades-real-money"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class RiskLimits:
    """Hard caps enforced before any order leaves the process."""

    max_position_notional: float = 1_000.0
    max_total_notional: float = 5_000.0
    max_open_positions: int = 5
    max_daily_loss_pct: float = 2.0
    max_orders_per_day: int = 20
    symbol_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct >= 100:
            raise ConfigError("max_daily_loss_pct must be between 0 and 100")
        if self.max_position_notional <= 0:
            raise ConfigError("max_position_notional must be positive")
        if self.max_total_notional < self.max_position_notional:
            raise ConfigError("max_total_notional must be >= max_position_notional")
        if not self.symbol_allowlist:
            raise ConfigError(
                "symbol_allowlist must not be empty -- an empty allowlist would "
                "permit trading any symbol the strategy happens to emit"
            )


@dataclass(frozen=True)
class Config:
    mode: str
    symbols: tuple[str, ...]
    strategy: str
    strategy_params: dict = field(default_factory=dict)
    timeframe: str = "1Day"
    poll_seconds: int = 60
    risk: RiskLimits = field(default_factory=RiskLimits)
    state_path: Path = Path("state/tradezbotz.db")
    kill_switch_path: Path = Path("KILL")
    api_key: str = ""
    api_secret: str = ""

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def endpoint(self) -> str:
        return LIVE_ENDPOINT if self.is_live else PAPER_ENDPOINT


def _require_live_acknowledgement() -> None:
    """Enforce the three-signal gate for live trading.

    Signal 1 is `mode: live` in the config file (checked by the caller).
    Signal 2 is the TRADEZBOTZ_ALLOW_LIVE environment variable.
    Signal 3 is a separate key pair, so paper keys cannot silently reach
    the live endpoint.
    """
    ack = os.environ.get("TRADEZBOTZ_ALLOW_LIVE", "")
    if ack != LIVE_ACK:
        raise ConfigError(
            "Refusing to start in live mode.\n"
            "config.yaml requests mode: live, but the environment does not arm it.\n"
            f"To proceed you must set TRADEZBOTZ_ALLOW_LIVE={LIVE_ACK}\n"
            "Do not do this until the strategy has run on paper long enough to "
            "trust its drawdown behaviour."
        )


def load(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No config file at {path}. Copy config.example.yaml first.")

    raw = yaml.safe_load(path.read_text()) or {}

    mode = str(raw.get("mode", "paper")).lower()
    if mode not in {"paper", "live"}:
        raise ConfigError(f"mode must be 'paper' or 'live', got {mode!r}")
    if mode == "live":
        _require_live_acknowledgement()

    symbols = tuple(str(s).upper() for s in raw.get("symbols", []))
    if not symbols:
        raise ConfigError("config must list at least one symbol")

    risk_raw = dict(raw.get("risk", {}))
    allowlist = risk_raw.pop("symbol_allowlist", None) or symbols
    risk = RiskLimits(
        symbol_allowlist=tuple(str(s).upper() for s in allowlist),
        **risk_raw,
    )

    unknown = set(symbols) - set(risk.symbol_allowlist)
    if unknown:
        raise ConfigError(
            f"symbols {sorted(unknown)} are not in risk.symbol_allowlist"
        )

    # Live and paper credentials are read from different variables on purpose.
    prefix = "ALPACA_LIVE" if mode == "live" else "ALPACA_PAPER"
    key = os.environ.get(f"{prefix}_API_KEY", "")
    secret = os.environ.get(f"{prefix}_API_SECRET", "")
    if not key or not secret:
        raise ConfigError(
            f"Missing credentials: set {prefix}_API_KEY and {prefix}_API_SECRET. "
            "See .env.example."
        )

    return Config(
        mode=mode,
        symbols=symbols,
        strategy=str(raw.get("strategy", "sma_cross")),
        strategy_params=dict(raw.get("strategy_params", {})),
        timeframe=str(raw.get("timeframe", "1Day")),
        poll_seconds=int(raw.get("poll_seconds", 60)),
        risk=risk,
        state_path=Path(raw.get("state_path", "state/tradezbotz.db")),
        kill_switch_path=Path(raw.get("kill_switch_path", "KILL")),
        api_key=key,
        api_secret=secret,
    )
