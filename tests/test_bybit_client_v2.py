import pytest
from unittest.mock import MagicMock, patch
from src.bybit_client import BybitClient

@pytest.fixture
def mock_session():
    with patch('src.bybit_client.HTTP') as MockHTTP:
        yield MockHTTP.return_value

def test_open_trade(mock_session):
    client = BybitClient(api_key="test", api_secret="test")
    client.session.place_order.return_value = {'retCode': 0, 'result': {'orderId': '123'}}

    # Original args
    client.open_trade("BTCUSDT", "Buy", 0.1, "Limit", 50000)

    # Note: open_trade logic uses keyword args now?
    # No, signature is def open_trade(self, symbol, side, qty, order_type="Market", price=None, category="linear", stop_loss=None, take_profit=None, trailing_stop=None)
    # The existing test call: open_trade("BTCUSDT", "Buy", 0.1, "Limit", 50000) matches first 5 args.

    client.session.place_order.assert_called_with(
        category="linear",
        symbol="BTCUSDT",
        side="Buy",
        orderType="Limit",
        qty="0.1",
        price="50000"
    )

def test_open_trade_with_sl_tp(mock_session):
    client = BybitClient(api_key="test", api_secret="test")
    client.session.place_order.return_value = {'retCode': 0, 'result': {'orderId': '124'}}

    # Use keywords for clarity
    client.open_trade(
        symbol="ETHUSDT",
        side="Sell",
        qty=1.0,
        order_type="Market",
        stop_loss=2000.0,
        take_profit=1800.0,
        trailing_stop=50.0
    )

    client.session.place_order.assert_called_with(
        category="linear",
        symbol="ETHUSDT",
        side="Sell",
        orderType="Market",
        qty="1.0",
        stopLoss="2000.0",
        takeProfit="1800.0",
        trailingStop="50.0"
    )

def test_get_orderbook(mock_session):
    client = BybitClient(api_key="test", api_secret="test")
    client.session.get_orderbook.return_value = {'retCode': 0, 'result': {'b': [], 'a': []}}

    client.get_orderbook("BTCUSDT", limit=20)

    client.session.get_orderbook.assert_called_with(
        category="linear",
        symbol="BTCUSDT",
        limit=20
    )
