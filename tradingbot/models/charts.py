import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_trades(account, full_data):
    # Group trades by stock
    trades_by_stock = {}
    for trade in account.sold_trades:
        stock = trade["stock"]
        if stock not in trades_by_stock:
            trades_by_stock[stock] = []
        trades_by_stock[stock].append(trade)

    for stock, trades in trades_by_stock.items():
        print(f"\n📈 Plotting trades for {stock}...")

        data = full_data[stock].copy()

        # Flatten multi-index if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index = pd.to_datetime(data.index)

        # Make sure all required columns exist
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            print(f"⚠️ Data for {stock} is missing required columns. Skipping.")
            continue

        # Create subplot with candlesticks + volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=(f"{stock} Candlestick Chart", "Volume")
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        ), row=1, col=1)

        # Volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='gray',
            opacity=0.5
        ), row=2, col=1)

        # Trade line traces
        trade_traces = []
        for i, trade in enumerate(trades):
            color = "green" if trade["sell_position"] > trade["buy_position"] else "red"
            trade_date = pd.to_datetime(trade["date"])
            trace = go.Scatter(
                x=[trade_date, trade_date],
                y=[trade["buy_position"], trade["sell_position"]],
                mode='lines+markers',
                name=f"{stock} Trade",
                line=dict(color=color, width=1),
                marker=dict(size=4),
                hovertext=f"Buy: {trade['buy_position']:.2f}<br>Sell: {trade['sell_position']:.2f}<br>P/L: {trade['profit/loss']:.2f}",
                showlegend=False,
                visible=True  # Initially visible
            )
            fig.add_trace(trace, row=1, col=1)
            trade_traces.append(len(fig.data) - 1)  # Save the index of this trace

        # Create toggle button
        buttons = [
            dict(
                label="Show Trades",
                method="update",
                args=[{"visible": [True] * len(fig.data)}]  # All traces visible
            ),
            dict(
                label="Hide Trades",
                method="update",
                args=[{
                    "visible": [
                        True if i < 2 else (i not in trade_traces)  # Keep candlestick + volume only
                        for i in range(len(fig.data))
                    ]
                }]
            )
        ]

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=1,
                y=1.1,
                showactive=True,
                buttons=buttons,
                xanchor="right",
                yanchor="top",
                pad={"r": 10, "t": 10}
            )],
            height=900,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        fig.show()
