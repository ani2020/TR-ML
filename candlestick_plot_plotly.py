import plotly.graph_objects as go
from datetime import datetime

def compute_trade_events(df):
    df = df.copy()

    df["position"] = df["signal"].shift(1).fillna(0)
    df["trade_signal"] = df["position"].diff().fillna(0)

    return df

def plot_candlestick_with_signals(df, title="Candlestick with Signals"):

    df = df.copy()

    # df = df.reset_index()

    # df = df.sort_values("date")
    # df = df.drop_duplicates(subset=["date"], keep="last")
    # df = df.set_index("date")

    # --- Buy/Sell markers ---
    #buy_signals = df[df["signal"] == 1]
    #sell_signals = df[df["signal"] == -1]

   # --- Compute trade events ---
    df["position"] = df["signal"].shift(1).fillna(0)
    df["trade_signal"] = df["position"].diff().fillna(0)

    # --- Trade markers ---
    long_entries = df[df["trade_signal"] == 1]
    long_exits = df[df["trade_signal"] == -1]

    short_entries = df[df["trade_signal"] == -2]
    short_exits = df[df["trade_signal"] == 2]

    #print(df.info())

    # Visualize using Plotly
    fig = go.Figure()

    #fig = go.Figure(data=[go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Nifty Close'))

    # Add buy signals
    fig.add_trace(go.Scatter(x=long_entries['date'], y=long_entries['close'], mode='markers', 
                             marker=dict(symbol='star-triangle-up', size=10, color='green'), name='long_entries'))
    fig.add_trace(go.Scatter(x=short_entries['date'], y=short_entries['close'], mode='markers', 
                             marker=dict(symbol='star-triangle-down', size=15, color='darkgreen'), name='flip-short_entries-long-exit'))
    fig.add_trace(go.Scatter(x=short_entries['date'], y=short_entries['close'], mode='markers', 
                             marker=dict(symbol='triangle-down-open', size=10, color='skyblue'), name='short_entries'))
    
    # Add sell signals
    fig.add_trace(go.Scatter(x=long_exits['date'], y=long_exits['close'], mode='markers', 
                             marker=dict(symbol='star-triangle-down', size=10, color='red'), name='long_exits'))
    fig.add_trace(go.Scatter(x=short_exits['date'], y=short_exits['close'], mode='markers', 
                             marker=dict(symbol='star-triangle-up', size=15, color='pink'), name='flip-short_exits-long-entry'))
    fig.add_trace(go.Scatter(x=short_exits['date'], y=short_exits['close'], mode='markers', 
                             marker=dict(symbol='triangle-up-open', size=10, color='cyan'), name='short_exits'))
    
    # Update layout with the new specifications
    fig.update_layout(
        title='Nifty 50 HMM Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis=dict(type='category')
    )
    
    # Show the plot
    fig.show()

    # print(df)

    # print(buy_signals)
    # print(sell_signals)