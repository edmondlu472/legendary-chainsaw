"""Platform configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class IBKRConfig(BaseSettings):
    model_config = {"env_prefix": "IBKR_"}

    host: str = "127.0.0.1"
    port: int = 7497  # 7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False


class PolygonConfig(BaseSettings):
    model_config = {"env_prefix": "POLYGON_"}

    api_key: str = ""
    base_url: str = "https://api.polygon.io"


class RiskConfig(BaseSettings):
    model_config = {"env_prefix": "RISK_"}

    max_position_pct: float = Field(default=0.05, description="Max single position as % of portfolio")
    max_portfolio_delta: float = Field(default=500.0, description="Max absolute portfolio delta")
    max_portfolio_vega: float = Field(default=10000.0, description="Max absolute portfolio vega")
    max_daily_drawdown_pct: float = Field(default=0.03, description="Kill switch: max daily drawdown %")
    max_margin_utilization: float = Field(default=0.70, description="Max margin usage before halting")
    max_correlated_positions: int = Field(default=5, description="Max positions in correlated assets")


class PlatformConfig(BaseSettings):
    model_config = {"env_prefix": "PLATFORM_"}

    mode: str = Field(default="paper", description="Trading mode: paper | live")
    log_level: str = "INFO"
    db_url: str = "sqlite:///chainsaw.db"

    ibkr: IBKRConfig = IBKRConfig()
    polygon: PolygonConfig = PolygonConfig()
    risk: RiskConfig = RiskConfig()


def load_config() -> PlatformConfig:
    return PlatformConfig()
