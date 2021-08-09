import tushare as ts


def get_stock_data():
    """
    获取指定股票代码历史记录
    :return:
    """
    sc = input('Please Enter Stock code: ')
    df = input('Date from xxxx-xx-xx: ')
    dt = input('Date from xxxx-xx-xx: ')
    data = ts.get_k_data(sc, ktype='D', start=df, end=dt)
    data_path = "./data/{}.csv".format(sc)
    data.to_csv(data_path)


get_stock_data()