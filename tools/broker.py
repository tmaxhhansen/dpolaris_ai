"""
Broker Integrations for dPolaris

Connects to brokerage accounts for:
- Real-time quotes and streaming data
- Options chain data with live Greeks
- Account balances and positions
- (Optional) Order placement

Supported brokers:
- E*Trade (official API - requires developer access)
- Webull (unofficial library)
- Interactive Brokers (ib_insync)
- Alpaca (free, paper trading)
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, AsyncIterator
import logging

logger = logging.getLogger("dpolaris.tools.broker")


class BrokerBase(ABC):
    """Base class for broker integrations"""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get real-time quote"""
        pass

    @abstractmethod
    async def get_options_chain(self, symbol: str, expiration: str = None) -> Optional[dict]:
        """Get options chain"""
        pass

    @abstractmethod
    async def get_account_info(self) -> Optional[dict]:
        """Get account balance and info"""
        pass

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Get current positions"""
        pass


# ==================== Webull Integration ====================

class WebullBroker(BrokerBase):
    """
    Webull integration using unofficial webull library.

    Install: pip install webull

    Note: This uses unofficial APIs and may break.
    Use at your own risk.
    """

    def __init__(self, email: str = None, password: str = None, device_id: str = None):
        self.email = email
        self.password = password
        self.device_id = device_id
        self.client = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Webull"""
        try:
            from webull import webull

            self.client = webull()

            if self.email and self.password:
                # Login with credentials
                self.client.login(self.email, self.password, self.device_id)
                logger.info("Connected to Webull with authentication")
            else:
                # Anonymous access (limited data)
                logger.info("Connected to Webull (anonymous - limited data)")

            self.connected = True
            return True

        except ImportError:
            logger.error("webull library not installed. Run: pip install webull")
            return False
        except Exception as e:
            logger.error(f"Webull connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Webull"""
        if self.client:
            try:
                self.client.logout()
            except:
                pass
        self.connected = False

    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get real-time quote from Webull"""
        if not self.client:
            return None

        try:
            quote = self.client.get_quote(symbol)

            if not quote:
                return None

            return {
                "symbol": symbol,
                "price": float(quote.get("close", 0)),
                "open": float(quote.get("open", 0)),
                "high": float(quote.get("high", 0)),
                "low": float(quote.get("low", 0)),
                "volume": int(quote.get("volume", 0)),
                "change": float(quote.get("change", 0)),
                "change_percent": float(quote.get("changeRatio", 0)) * 100,
                "bid": float(quote.get("bid", 0)),
                "ask": float(quote.get("ask", 0)),
                "timestamp": datetime.now().isoformat(),
                "source": "webull",
            }
        except Exception as e:
            logger.error(f"Webull quote error for {symbol}: {e}")
            return None

    async def get_options_chain(self, symbol: str, expiration: str = None) -> Optional[dict]:
        """Get options chain from Webull"""
        if not self.client:
            return None

        try:
            # Get available expirations
            expirations = self.client.get_options_expiration_dates(symbol)

            if not expirations:
                return None

            # Use specified or first expiration
            exp_date = expiration or expirations[0].get("date")

            # Get options data
            options_data = self.client.get_options(symbol, exp_date)

            if not options_data:
                return None

            calls = []
            puts = []

            for opt in options_data:
                option_dict = {
                    "strike": float(opt.get("strikePrice", 0)),
                    "last_price": float(opt.get("close", 0)),
                    "bid": float(opt.get("bid", 0)),
                    "ask": float(opt.get("ask", 0)),
                    "volume": int(opt.get("volume", 0)),
                    "open_interest": int(opt.get("openInterest", 0)),
                    "implied_volatility": float(opt.get("impVol", 0)),
                    "delta": float(opt.get("delta", 0)),
                    "gamma": float(opt.get("gamma", 0)),
                    "theta": float(opt.get("theta", 0)),
                    "vega": float(opt.get("vega", 0)),
                }

                if opt.get("direction") == "call":
                    calls.append({**option_dict, "type": "call"})
                else:
                    puts.append({**option_dict, "type": "put"})

            return {
                "symbol": symbol,
                "expiration": exp_date,
                "expirations_available": [e.get("date") for e in expirations],
                "calls": calls,
                "puts": puts,
                "timestamp": datetime.now().isoformat(),
                "source": "webull",
            }

        except Exception as e:
            logger.error(f"Webull options error for {symbol}: {e}")
            return None

    async def get_account_info(self) -> Optional[dict]:
        """Get account info from Webull"""
        if not self.client or not self.connected:
            return None

        try:
            account = self.client.get_account()

            return {
                "total_value": float(account.get("totalMarketValue", 0)),
                "cash": float(account.get("cashBalance", 0)),
                "buying_power": float(account.get("dayBuyingPower", 0)),
                "unrealized_pnl": float(account.get("unrealizedProfitLoss", 0)),
                "day_pnl": float(account.get("dayProfitLoss", 0)),
                "source": "webull",
            }
        except Exception as e:
            logger.error(f"Webull account error: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        """Get positions from Webull"""
        if not self.client or not self.connected:
            return []

        try:
            positions = self.client.get_positions()

            return [
                {
                    "symbol": p.get("ticker", {}).get("symbol"),
                    "quantity": float(p.get("position", 0)),
                    "avg_cost": float(p.get("costPrice", 0)),
                    "current_price": float(p.get("lastPrice", 0)),
                    "market_value": float(p.get("marketValue", 0)),
                    "unrealized_pnl": float(p.get("unrealizedProfitLoss", 0)),
                    "unrealized_pnl_percent": float(p.get("unrealizedProfitLossRate", 0)) * 100,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Webull positions error: {e}")
            return []

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[dict]:
        """Stream real-time quotes (polling-based for Webull)"""
        while True:
            for symbol in symbols:
                quote = await self.get_quote(symbol)
                if quote:
                    yield quote
            await asyncio.sleep(1)  # 1 second polling


# ==================== E*Trade Integration ====================

class ETradeBroker(BrokerBase):
    """
    E*Trade integration using official API.

    Requires:
    1. E*Trade developer account: https://developer.etrade.com
    2. Consumer key and secret
    3. OAuth authentication flow

    Install: pip install pyetrade
    """

    def __init__(
        self,
        consumer_key: str = None,
        consumer_secret: str = None,
        sandbox: bool = True,
    ):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox = sandbox
        self.session = None
        self.accounts = None
        self.connected = False

    async def connect(self) -> bool:
        """
        Connect to E*Trade.

        Note: E*Trade requires OAuth flow which needs user interaction
        for the first connection. Consider using a token cache.
        """
        try:
            import pyetrade

            # OAuth flow
            oauth = pyetrade.ETradeOAuth(self.consumer_key, self.consumer_secret)
            request_token, request_token_secret = oauth.get_request_token()

            # Get authorization URL (user needs to visit this)
            auth_url = oauth.get_authorize_url(request_token)
            logger.info(f"Please authorize at: {auth_url}")

            # In production, you'd cache tokens and handle this better
            verifier = input("Enter verifier code: ")

            access_token, access_token_secret = oauth.get_access_token(verifier)

            # Create authenticated session
            self.session = pyetrade.ETradeAccounts(
                self.consumer_key,
                self.consumer_secret,
                access_token,
                access_token_secret,
                dev=self.sandbox,
            )

            self.connected = True
            logger.info("Connected to E*Trade")
            return True

        except ImportError:
            logger.error("pyetrade library not installed. Run: pip install pyetrade")
            return False
        except Exception as e:
            logger.error(f"E*Trade connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from E*Trade"""
        self.session = None
        self.connected = False

    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get quote from E*Trade"""
        if not self.session:
            return None

        try:
            import pyetrade

            market = pyetrade.ETradeMarket(
                self.consumer_key,
                self.consumer_secret,
                # tokens would be stored
            )

            quote_data = market.get_quote([symbol])

            if not quote_data:
                return None

            q = quote_data.get("QuoteResponse", {}).get("QuoteData", [{}])[0]
            all_data = q.get("All", {})

            return {
                "symbol": symbol,
                "price": float(all_data.get("lastTrade", 0)),
                "bid": float(all_data.get("bid", 0)),
                "ask": float(all_data.get("ask", 0)),
                "volume": int(all_data.get("totalVolume", 0)),
                "timestamp": datetime.now().isoformat(),
                "source": "etrade",
            }

        except Exception as e:
            logger.error(f"E*Trade quote error: {e}")
            return None

    async def get_options_chain(self, symbol: str, expiration: str = None) -> Optional[dict]:
        """Get options chain from E*Trade"""
        # Implement similar to Webull
        logger.warning("E*Trade options chain not fully implemented")
        return None

    async def get_account_info(self) -> Optional[dict]:
        """Get E*Trade account info"""
        if not self.session:
            return None

        try:
            accounts = self.session.list_accounts()
            # Parse and return account info
            return {"accounts": accounts, "source": "etrade"}
        except Exception as e:
            logger.error(f"E*Trade account error: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        """Get E*Trade positions"""
        if not self.session:
            return []

        try:
            # Get positions for each account
            positions = []
            # Implementation would iterate through accounts
            return positions
        except Exception as e:
            logger.error(f"E*Trade positions error: {e}")
            return []


# ==================== Interactive Brokers ====================

class IBKRBroker(BrokerBase):
    """
    Interactive Brokers integration using ib_insync.

    Requires:
    1. IBKR account (paper or live)
    2. TWS or IB Gateway running
    3. API connections enabled

    Install: pip install ib_insync
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port  # 7497 for TWS paper, 7496 for live
        self.client_id = client_id
        self.ib = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to TWS/IB Gateway"""
        try:
            from ib_insync import IB

            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)

            self.connected = self.ib.isConnected()
            if self.connected:
                logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return self.connected

        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib:
            self.ib.disconnect()
        self.connected = False

    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get real-time quote from IBKR"""
        if not self.ib or not self.connected:
            return None

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(1)  # Wait for data

            return {
                "symbol": symbol,
                "price": ticker.last or ticker.close,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "volume": ticker.volume,
                "timestamp": datetime.now().isoformat(),
                "source": "ibkr",
            }

        except Exception as e:
            logger.error(f"IBKR quote error: {e}")
            return None

    async def get_options_chain(self, symbol: str, expiration: str = None) -> Optional[dict]:
        """Get options chain from IBKR"""
        if not self.ib or not self.connected:
            return None

        try:
            from ib_insync import Stock

            stock = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(stock)

            chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)

            # Get option contracts
            # This is simplified - full implementation would be more complex
            return {
                "symbol": symbol,
                "chains": chains,
                "source": "ibkr",
            }

        except Exception as e:
            logger.error(f"IBKR options error: {e}")
            return None

    async def get_account_info(self) -> Optional[dict]:
        """Get IBKR account info"""
        if not self.ib or not self.connected:
            return None

        try:
            account_values = self.ib.accountValues()

            info = {}
            for av in account_values:
                if av.tag == "NetLiquidation":
                    info["total_value"] = float(av.value)
                elif av.tag == "TotalCashValue":
                    info["cash"] = float(av.value)
                elif av.tag == "UnrealizedPnL":
                    info["unrealized_pnl"] = float(av.value)

            info["source"] = "ibkr"
            return info

        except Exception as e:
            logger.error(f"IBKR account error: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        """Get IBKR positions"""
        if not self.ib or not self.connected:
            return []

        try:
            positions = self.ib.positions()

            return [
                {
                    "symbol": p.contract.symbol,
                    "quantity": p.position,
                    "avg_cost": p.avgCost,
                    "contract_type": p.contract.secType,
                }
                for p in positions
            ]

        except Exception as e:
            logger.error(f"IBKR positions error: {e}")
            return []

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[dict]:
        """Stream real-time quotes from IBKR"""
        if not self.ib or not self.connected:
            return

        from ib_insync import Stock

        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        self.ib.qualifyContracts(*contracts)

        tickers = [self.ib.reqMktData(c) for c in contracts]

        while True:
            for ticker, symbol in zip(tickers, symbols):
                if ticker.last:
                    yield {
                        "symbol": symbol,
                        "price": ticker.last,
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                        "timestamp": datetime.now().isoformat(),
                    }
            await asyncio.sleep(0.1)


# ==================== Broker Factory ====================

def get_broker(
    broker_name: str,
    **kwargs,
) -> Optional[BrokerBase]:
    """
    Factory function to get broker instance.

    Args:
        broker_name: 'webull', 'etrade', 'ibkr', 'alpaca'
        **kwargs: Broker-specific configuration

    Returns:
        Broker instance
    """
    brokers = {
        "webull": WebullBroker,
        "etrade": ETradeBroker,
        "ibkr": IBKRBroker,
    }

    broker_class = brokers.get(broker_name.lower())
    if broker_class:
        return broker_class(**kwargs)

    logger.error(f"Unknown broker: {broker_name}")
    return None


# ==================== Unified Data Service ====================

class UnifiedDataService:
    """
    Unified service that tries multiple data sources.

    Priority:
    1. Connected broker (real-time)
    2. Yahoo Finance (fallback)
    """

    def __init__(self, broker: Optional[BrokerBase] = None):
        self.broker = broker
        self.fallback_service = None

    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get quote from best available source"""
        # Try broker first
        if self.broker and self.broker.connected:
            quote = await self.broker.get_quote(symbol)
            if quote:
                return quote

        # Fallback to Yahoo
        from .market_data import fetch_quote
        return await fetch_quote(symbol)

    async def get_options_chain(self, symbol: str, expiration: str = None) -> Optional[dict]:
        """Get options chain from best available source"""
        if self.broker and self.broker.connected:
            chain = await self.broker.get_options_chain(symbol, expiration)
            if chain:
                return chain

        from .market_data import fetch_options_chain
        return await fetch_options_chain(symbol, expiration)
