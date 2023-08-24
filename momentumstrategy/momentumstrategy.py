"""Momentum Strategy"""
import time
from typing import Dict

import matplotlib.pyplot as plt  # noqa: F401
import pandas as pd
from strategybacktest import Backtest, BacktestAnalysis, Portfolio, Strategy


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy.

    TODO: FILL THIS DOCSTRING

    Args:
        weights_df: Dataframe of initial weights for each asset.
        prices_df: Dataframe of daily prices for each asset. We ensure only new
        day prices are used to avoid look ahead bias. Hence, the price record
        etc. below.
    """

    def __init__(self) -> None:
        self._prices_record = pd.DataFrame()
        self._current_weights = {}
        self._months_set = set()
        self._current_month = None

    def __call__(
        self, ts: pd.Timestamp, prices: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Rebalance portfolio for each new timestamp.

        Args:
            ts: Current timestamp.
            prices: Dataframe of prices for each asset on ts.

        Returns:
            Portfolio weights.
        """
        # Update historical prices
        self._prices_record = pd.concat([self._prices_record, prices])
        # Add latest month to set
        self._months_set.add(ts.month)

        # For the first 6 months, return the no weights.
        # Check if we have 6 full calendar months of data in the price record.
        # NOTE: Ignore the first month as it may not be complete.
        if len(self._months_set) <= 6:
            self._current_month = ts.month
            return self._current_weights

        elif self._current_month == ts.month:
            # If we are still in the same month, return the same weights.
            # Rebalance monthly.
            self._current_month = ts.month
            return self._current_weights

        else:
            # Discarding the previous month, we take average daily returns for
            # each month.
            # 1. Cut the price record to the last 6 months excluding the
            # previous month (so 5 months in length).
            # Calculate the first timestamp of the 6 month period, this is the
            # first day of the month THAT IS IN self._prices_record 6 months
            # ago.
            monthly_first_df = self._prices_record.groupby(
                pd.Grouper(freq="M")
            ).nth(0)
            monthly_last_df = self._prices_record.groupby(
                pd.Grouper(freq="M")
            ).nth(-1)
            start_ts = monthly_first_df.index[-6]
            end_ts = monthly_last_df.index[-2]
            window_prices = self._prices_record.loc[start_ts:end_ts][:-1]
            # 2. Calculate average daily returns for the 5 months.
            # (This is appropriate because we rebalance daily.)
            returns_df = window_prices.pct_change().dropna().mean()

            # Pick top 10% and bottom 10% of assets as winners and losers
            # respectively using
            winners = returns_df.nlargest(int(len(returns_df) * 0.1))
            losers = returns_df.nsmallest(int(len(returns_df) * 0.1))
            # Calculate the weights (equal long/short weights to all
            # winners/losers)
            weight = 0.5 / len(winners)
            # Update current weights
            self._current_weights = {winner: weight for winner in winners.index}
            self._current_weights.update(
                {loser: -weight for loser in losers.index}
            )

        self._current_month = ts.month
        return self._current_weights


def run_backtest(
    initial_capital: float,
    risk_free_rate: float,
    transaction_cost: float,
    plot: bool,
    save_plots: bool = False,
) -> None:
    """
    Run the backtest.

    Args:
        initial_capital: Initial capital to invest.
        risk_free_rate: Risk free rate.
        transaction_cost: Percentage transaction cost per trade.
        plot: Plot backtest results.
        save_plots: Save backtest plots. Defaults to False.
    """
    data_filepath = ".data/S&P_5yr.csv"
    # Collect data.
    # I can plot here to check for outliers. There are none.
    prices_df = (
        pd.read_csv(data_filepath, index_col=0, parse_dates=True)
        .dropna(how="all")
        .drop_duplicates()
    )
    nan_stats_df = prices_df.isna().sum()
    # Make list of stocks with large number of NaNs (over 10% of values missing)
    invalid_tickers = list(
        nan_stats_df[nan_stats_df > 0.1 * len(prices_df)].index
    )
    # Drop these stocks from the dataframe
    prices_df = prices_df.drop(columns=invalid_tickers)
    # Fill remaining NaNs with mean of price of stock
    prices_df = prices_df.fillna(prices_df.mean())

    timestamps = prices_df.index.tolist()
    # NOTE: Asset universe for future versions
    # asset_universe = list(prices_df.columns)

    # Initialise strategy
    strategy = MomentumStrategy()

    # Initialise portfolio
    portfolio = Portfolio(
        initial_capital=initial_capital,
        price_data_source=prices_df,
        transaction_cost=transaction_cost,
    )
    # Initialise backtest
    backtest = Backtest(
        strategy=strategy,
        timestamps=timestamps,
        portfolio=portfolio,
        price_data_source=prices_df,
    )
    # Run backtest
    backtest.run_backtest()

    # Run analysis
    analyser = BacktestAnalysis(
        backtest=backtest, risk_free_rate=risk_free_rate
    )
    analyser.compute_stats()

    # Plot results
    if plot:
        analyser.plot(save=save_plots)
        analyser.underwater_plot(save=save_plots)
        analyser.volatility_plot(save=save_plots)

    # Save results to excel
    analyser.output_to_excel(
        filepath=f"output/summary_ic{initial_capital}_tc{transaction_cost}"
        f"_rf{risk_free_rate}.xlsx"
    )


if __name__ == "__main__":
    # Run backtest(s) and produce time-series and summary results as excel
    # files.
    initial_capital = 100000

    start_time = time.time()

    run_backtest(
        initial_capital=initial_capital,
        risk_free_rate=0,
        transaction_cost=0.003,
        plot=True,
        save_plots=False,
    )

    print(f"--- {time.time() - start_time} seconds ---")
