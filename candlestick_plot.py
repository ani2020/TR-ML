import mplfinance as mpf
import pandas as pd


def plot_candlestick_with_signals(df, title="Candlestick with Signals"):

    df = df.copy()


    # --- mplfinance requires OHLC ---
    # For now we fake OHLC using close (until real OHLC added)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "date": "Date"
    }, inplace=True)

    # # --- Ensure datetime index ---

    #df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")

    # --- Buy/Sell markers ---
    #buy_signals = df[df["signal"] == 1]
    #sell_signals = df[df["signal"] == -1]
    buy_series = df["Close"].where(df["signal"] == 1)
    sell_series = df["Close"].where(df["signal"] == -1)

    print(df)

    print(f"length of olhc data: {len(df)}")
    print(f"length of buy signal series: {len(buy_series)}")
    print(f"length of sell signal series: {len(sell_series)}")

    print(f"length of buy signal series: {buy_series}")
    print(f"length of sell signal series: {sell_series}")


    apds = []

    apds.append(
        mpf.make_addplot(
            buy_series,
            type="scatter",
            markersize=100,
            marker="^"
        )
    )

    apds.append(
        mpf.make_addplot(
            sell_series,
            type="scatter",
            markersize=100,
            marker="v"
        )
    )

    # --- Regime shading ---
    # We create color bands manually
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        inherit=True
    )

    style = mpf.make_mpf_style(marketcolors=mc)

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=apds,
        title=title,
        returnfig=True,
        volume=False
    )

    ax = axes[0]

    # --- Background shading for regimes ---
    if "regime" in df.columns:
        regime_colors = {
            "bull": "green",
            "bear": "red",
            "sideways": "gray"
        }

        prev_regime = None
        start_idx = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            regime = row["regime"]

            if regime != prev_regime:
                if prev_regime is not None:
                    ax.axvspan(
                        df.index[start_idx],
                        df.index[i],
                        color=regime_colors.get(prev_regime, "gray"),
                        alpha=0.1
                    )
                start_idx = i
                prev_regime = regime

        # last segment
        ax.axvspan(
            df.index[start_idx],
            df.index[-1],
            color=regime_colors.get(prev_regime, "gray"),
            alpha=0.1
        )

    mpf.show()