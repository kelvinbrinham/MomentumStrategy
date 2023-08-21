"""Momentum Strategy"""

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
        self._initial = True
        self._second = True

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

        if ts in self.weights_df.index:
            # For the first two months, return the default weights.
            if self._initial or self._second:
                self._current_weights = {}

            else:
                # Calculate the returns for each asset
                month_prices = self._prices_record[
                    ts
                    - pd.DateOffset(months=2) : ts
                    - pd.DateOffset(months=1, days=1)
                ]
                returns_df = month_prices.pct_change().dropna().mean()
                winner = returns_df.idxmax()
                loser = returns_df.idxmin()
                self._current_weights = {winner: 0.5, loser: -0.5}

            if not self._initial:
                self._second = False
            self._initial = False

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
    prices_df = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
    print(prices_df.head())
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
        timestamps=prices_df.index.values,
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

    run_backtest(
        initial_capital=initial_capital,
        risk_free_rate=0,
        transaction_cost=0.003,
        plot=True,
        save_plots=True,
    )
