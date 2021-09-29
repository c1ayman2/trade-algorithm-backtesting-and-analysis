"""#



"""

import copy
import statistics
import math
import finnhub
import MyAPIKeyForFinnhub as keys
import time
from dataclasses import dataclass
import datetime
from statistics import mean

import pandas.io.pytables
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import talib as ta
import numpy as np
import pprint as pp

pd.set_option("display.max_columns", 99)

finnhub_client = finnhub.Client(api_key=keys.api_key)


def popup_candles(df):
    hrt = [time.ctime(int(x)) for x in df["t"]]
    fig = go.Figure(data=[go.Candlestick(x=hrt,
                                         open=df["o"],
                                         high=df["h"],
                                         low=df["l"],
                                         close=df["c"])])
    fig.show()
    print(df)
    return fig


def avg(*args):  # returns the average
    return statistics.mean(args)


def last_60_days(endtime=pd.Timestamp.now()):  # returns a delta of two dates (start, end) for the past 60 days
    # looking back at this, what the fuck was i trying to do.
    # "2020-04-06 20:00:00"
    return [pd.Timestamp(endtime) - pd.Timedelta(days=60), pd.Timestamp(endtime)]


def to_seconds(args):
    return [math.trunc(ar.timestamp()) for ar in args]


def get_bin_candles_df(start_date=1590988249, end_date=1591852249):
    result = finnhub_client.crypto_candles('BINANCE:BTCUSDT', 'D', start_date, end_date)
    return pd.DataFrame(result)


def save_dataframe(dataframe):
    dataframe.to_pickle("api_cache")
    return


def load_dataframe():
    return pd.read_pickle("api_cache")


@dataclass
class Trade:  # contains historical information about prior trades
    # yeah, you'd think id want to make like, a dictionary or some sort of basic structure, but i want coupled functionality and the ability to openly move it explicitly

    buy_price: int = None
    start_date: int = None

    sell_price: int = None
    quantity: float = None
    close_date: int = None
    close_profit: float = None
    profitability: float = None  # should only be read/written after close

    def print(self):
        print(("BP ", self.buy_price,
               ",SP ", self.sell_price,
               ",SD ", self.start_date,
               ",CD ", self.close_date,
               ",Q ", self.quantity,
               ",CP ", self.close_profit,
               ",P ", self.profitability,
               ))

    def trade_age(self, ctime):
        if self.close_date:
            return self.close_date - self.start_date
        return ctime - self.start_date

    def open_trade(self, buy_price, budget, ctime):
        self.buy_price = buy_price
        self.start_date = ctime
        self.quantity = budget / buy_price

    def get_value(self, current_price):
        if self.close_date:
            return self.quantity * self.sell_price
        return self.quantity * current_price

    def how_profitable(self, current_price):  # returns the nX multiple of on a relativistic scale of how profitable the trade is.
        # a value less than 1 is not profitable
        if self.close_date is not None:
            return self.sell_price / self.buy_price
        return current_price / self.buy_price

    def close_trade(self, sell_price, ctime):
        self.sell_price = sell_price
        self.close_profit = (self.buy_price * self.quantity) - (self.quantity * sell_price)
        self.profitability = self.how_profitable(sell_price)
        self.close_date = ctime
        return


class Trader:  # superclass that controls the methods of operations. this is where analytics will be collected

    o_trades: [Trade] = []  # open trades. only open trades have value
    c_trades: [Trade] = []  # closed trades
    df: pd.DataFrame
    perf_df: pd.DataFrame
    liquid_money_remaining: float = None
    initial_balance: float = None
    perf_data = {
        # i dont want some sort of fancy structure because i intend to turn this into a df at the end of trade phase
        # and i dont want to have to normalize anything to make it happen
        'profit': [float],
        'net_gain': [float],
        'balance': [float],
        'time': [int],
        'performance': [str],
        'total_age_s': [int]
    }

    def __init__(self, df, money: float = 15000):
        self.df = df  # i briefly considered making this a deep copy so i could modify it (you shouldn't) , but then i realized i could just make another dataframe after
        self.liquid_money_remaining = money
        self.initial_balance = money

    def buy(self, buy_price, budget,
            buy_time: int):  # create trade object and chuck in into o_trades, subtract from money
        if self.liquid_money_remaining < 1:
            return False

        if budget > self.liquid_money_remaining:
            budget = self.liquid_money_remaining
        trade = Trade()
        trade.open_trade(buy_price, budget, buy_time)
        self.o_trades.append(trade)
        self.liquid_money_remaining -= budget
        return True

    def sell(self, sell_price, ctime: int):  # move trade object to c_trades and add value to money
        if self.o_trades:
            sold_trade = self.o_trades.pop(0)
            sold_trade.close_trade(sell_price, ctime)
            self.c_trades.append(sold_trade)
            self.liquid_money_remaining += self.c_trades[-1].get_value(
                sell_price)  # you sold the trade, add it back to wallet
        return

    # analytics

    def calc_profit(self, current_price):
        # how much has been earned since (excluding)initial balance, including assets
        return (self.liquid_money_remaining + self.assess_worth(current_price)) - self.initial_balance

    def assess_worth(self, current_value):  # total up hard equity bound up in trades
        return sum(trade.get_value(current_value) for trade in self.o_trades)

    def assess_age(self, ctime):  # the total age of how long all the trades have been open.
        return sum(trade.trade_age(ctime) for trade in self.o_trades)

    def assess_performance(self, current_value):  # grade trading on criterions, runs at the end of a time cycle
        # each criterion gives +1 to ranking: average trade age, average balance growth, proportion of money in positive trades
        pass

    def popup_chart_balance(self):
        self.perf_df = pd.DataFrame.from_dict(self.perf_data)
        print("perf_df")
        print(self.perf_df)
        # linechart = px.line(x=self.perf_df["time"][1:], y=[self.perf_df["balance"][1:], self.perf_df["net_gain"][1:]+self.initial_balance])
        linechart = px.line(x=self.perf_df["time"][1:],
                            y=self.perf_df["balance"][1:])
        # linechart.add
        # go.scatter.Line(self.perf_data["balance"])
        linechart.show()
        return

    def popup_chart_profit(self):
        self.perf_df = pd.DataFrame.from_dict(self.perf_data)
        print("perf_df")
        # print(self.perf_df)
        # linechart = px.line(x=self.perf_df["time"][1:], y=[self.perf_df["balance"][1:], self.perf_df["net_gain"][1:]+self.initial_balance])
        linechart = px.line(x=self.perf_df["time"][1:],
                            y=self.perf_df["profit"][1:],self.df)

        # go.scatter.Line(self.perf_data["balance"])
        linechart.show()
        return

    def popup_chart_trade_history(self):
        pass

    def print_trade_history(self):
        print("open trades")
        for trade in self.o_trades:
            trade.print()
        print("closed trades")
        for trade in self.c_trades:
            trade.print()
        print(self.liquid_money_remaining)
        pp.pprint(self.perf_data)
        return

    def gather_data(self, ctime: int, current_value):
        self.perf_data['profit'].append(self.calc_profit(current_value))
        self.perf_data['net_gain'].append(self.assess_worth(current_value))
        self.perf_data['balance'].append(self.liquid_money_remaining)  # looks like i found a bug in pycharm?
        self.perf_data['time'].append(ctime)
        self.perf_data['performance'].append(self.assess_performance(current_value))
        self.perf_data['total_age_s'].append(self.assess_age(ctime))
        return


class AlgoNaive(Trader):  # child classes implement the how of operations.
    # notes:
    # not very good at taking profits.
    def __init__(self, df):
        super().__init__(df)

    def run_algo(self):
        # if there is a positive sum, buy
        # if there is a negative sum, sell
        for row in df.itertuples():
            thesum = np.nansum(row[8:])
            if thesum > 5:
                self.buy(avg(row.o, row.c), 55, row.t)
            elif thesum < -5:
                self.sell(avg(row.o, row.c), row.t)
            self.gather_data(row.t, avg(row.o, row.c))

        self.print_trade_history()
        self.popup_chart_profit()
        # print(row.o)
        return

    def algo(self, row):

        pass


class AlgoRanked(Trader):
    def algo(self):  # regardless of sum, ignore poor performing candle patterns
        pass

    def __init__(self, df):
        super().__init__(df)

class AlgoDCAProfit(Trader):

    def sell_at_profit(self, sell_price, ctime: int, ratio):
        for trade in self.o_trades:
            if trade.how_profitable(sell_price) > ratio:
                self.sell(sell_price, ctime)
        return

    def algo(self):  # continually purchase slowly, sell when you can wih at least N* profit
        # if there is a positive sum, buy
        # if there is a negative sum, sell
        for row in df.itertuples():
            thesum = np.nansum(row[8:])
            if thesum > 5:
                self.buy(avg(row.o, row.c), 55, row.t)
            elif thesum < -5:
                self.sell_at_profit(avg(row.o, row.c), row.t, 1.02)
            self.gather_data(row.t, avg(row.o, row.c))

        self.print_trade_history()
        self.popup_chart_profit()
        # print(row.o)
        return

    def __init__(self, df):
        super().__init__(df)

class AlgoDCA(Trader):
    def algo(self):  # continually purchase slowly, sell only when you can sell everything wih at least N* profit
        pass

    def __init__(self, df):
        super().__init__(df)


class AlgoDCAStopLoss(Trader):
    def algo(
            self):  # continually purchase slowly, sell when near a stop loss for each trade or when sees bearish symbols near n day high.
        pass

    def __init__(self, df):
        super().__init__(df)


class AlgoPicky(Trader):
    def algo(self):  # ignore all but 5*2 of the best signals
        pass

    def __init__(self, df):
        super().__init__(df)


class AlgoScared(Trader):
    def algo(self):  # Buy only on the strongest bullish patterns, sell on generally any negative pattern
        pass

    def __init__(self, df):
        super().__init__(df)


class AlgoProphet(Trader):
    def algo(self):  # similar to DCA, but only sell on 10x profit
        pass

    def __init__(self, df):
        super().__init__(df)


if __name__ == '__main__':

    candle_names = ta.get_function_groups()['Pattern Recognition']

    # print(to_seconds(last_60_days()))

    # popup_candles(get_bin_candles_df())
    # candles = get_bin_candles_df(*to_seconds(last_60_days()))
    # save_dataframe(candles)

    df = load_dataframe()
    for candle in candle_names:  # use TA-Lib to parse candles
        df[candle] = getattr(ta, candle)(df["o"], df["h"], df["l"], df["c"])

    print(df)
    print(df.columns)

    df.replace(0, np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True) ###### if you're going to stream, i recommend against this

    print("after dropna")
    print(df)

    print("class tests")
    #algoclass = AlgoNaive(df)
    #algoclass.run_algo()

    algoclass2 = AlgoDCAProfit(df)
    algoclass2.algo()

    # print(df.columns)
    # print(df[candle_names]['CDLLONGLEGGEDDOJI'].iloc[1])

    # popup_candles(df)

    # print(*last_60_days("2020-04-06 20:00:00"))
    # print(*last_60_days())
    pass
