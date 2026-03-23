import matplotlib.pyplot as plt

def plot_signals(df, title="Strategy Signals"):
    df = df.copy()
    
    plt.figure(figsize=(14, 7))

    # --- Color by regime ---
    if "regime" in df.columns:
        colors = {
            "bull": "green",
            "bear": "red",
            "sideways": "gray"
        }

        for regime, group in df.groupby("regime"):
            plt.plot(
                group["date"],
                group["close"],
                color=colors.get(regime, "black"),
                label=f"{regime} regime",
                alpha=0.6
            )
    else:
        plt.plot(df["date"], df["close"], color="black", label="Price")

    # --- Signals ---
    buy = df[df["signal"] == 1]
    sell = df[df["signal"] == -1]

    plt.scatter(buy["date"], buy["close"], marker="^", color="green", s=100)
    plt.scatter(sell["date"], sell["close"], marker="v", color="red", s=100)

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# def plot_signals(df, title="Strategy Signals"):
#     df = df.copy()

#     plt.figure(figsize=(14, 7))

#     # --- Price line ---
#     plt.plot(df["date"], df["close"], label="Price", color="black")

#     # --- Buy signals ---
#     buy_signals = df[df["signal"] == 1]
#     plt.scatter(
#         buy_signals["date"],
#         buy_signals["close"],
#         marker="^",
#         color="green",
#         label="Buy",
#         s=100
#     )

#     # --- Sell signals ---
#     sell_signals = df[df["signal"] == -1]
#     plt.scatter(
#         sell_signals["date"],
#         sell_signals["close"],
#         marker="v",
#         color="red",
#         label="Sell",
#         s=100
#     )

#     plt.title(title)
#     plt.xlabel("Date")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.grid()

#     plt.xticks(rotation=45)

#     plt.tight_layout()
#     plt.show()