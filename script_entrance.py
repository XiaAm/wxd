"""
从指定文件夹里读取每个csv的某一天的数据，组成一个csv，输出到指定位置。

该csv用于atom，预测。

删除不必要的列。
新增一列id。
"""
from math import floor
import pandas as pd
import argparse
import os, sys
import datetime
from jqdatasdk import *
from shutil import copyfile

#auth('15355402003', '111111')
auth('15558150270', '111111')


def preprocess(df):
    df["code"] = df.apply(lambda row: str(int(row["code"])).zfill(6), axis=1)
    return df


def get_adj_code(code):
    """
    为股票代码添加沪、深标签
    """
    if code.find("0")==0 or code.find("3")==0:
        code_adj = code + ".XSHE"
    elif code.find("6") == 0:
        code_adj = code + ".XSHG"
    else:
        print("name %s not find, exit." %(code))
        sys.exit(-1)
    return code_adj


def read_source_csv(csv_file_path):
    """
    读取现在已经存在的沪深300股票csv文件
    考虑如果文件为空的情况。 
    """
    df = pd.read_csv( csv_file_path )
    df["code"] = df.apply( lambda row: str( row["code"] ).zfill( 6 ), axis=1 )
    df = df.sort_values( by=["date"], ascending=[True] )
    df.index = range( len( df ) )
    return df


def get_df_start_date(df):
    """
    获取csv文件最后一天数据的加一天，作为接口的开始时间
    """
    last_date = df.at[len( df ) - 1, "date"]
    last_date = datetime.datetime.strptime( last_date, "%Y-%m-%d" )
    start_date = last_date + datetime.timedelta( days=1 )
    start_date = start_date.strftime( "%Y-%m-%d" )
    last_date = last_date.strftime("%Y-%m-%d")
    return start_date, last_date


def get_price_df(code_adj, start_date, target_date):
    """
    获取价格接口
    """
    code = code_adj.split(".")[0]
    df_tmp = get_price( code_adj, start_date=start_date, end_date=target_date, frequency="daily",
                        fields=["open", "high", "low", "close", "volume", "money", "factor"], skip_paused=True )
    if len( df_tmp ) < 1:
        return df_tmp
    else:
        df_tmp = df_tmp.rename( columns={"money": "amount"} )
        df_tmp["date"] = df_tmp.index
        df_tmp["date"] = df_tmp.apply( lambda row: row["date"].strftime( "%Y-%m-%d" ), axis=1 )
        df_tmp["code"] = [code] * len( df_tmp )
        df_tmp["vwap"] = df_tmp.apply( lambda row: (row["high"] + row["low"] + row["close"]) / 3, axis=1 )
        return df_tmp.sort_values(by=["date"], ascending=[True])


def get_fundamentals_df(code_adj, date):
    """
    获取基本面接口
    """
    df_tojoin = pd.DataFrame()
    df_new = get_fundamentals( query( valuation ).filter( valuation.code == code_adj ), date )
    df_tojoin = pd.concat( [df_tojoin, df_new] )
    del df_tojoin["id"]
    del df_tojoin["code"]
    df_tojoin = df_tojoin.rename( columns={"day": "date", "market_cap": "cap"} )
    # cap的单位是 亿元，把它乘以1e8, https://www.joinquant.com/data/dict/fundamentals
    df_tojoin["cap"] = df_tojoin.apply( lambda row: row["cap"] * 1e8, axis=1 )
    return df_tojoin.sort_values(by=["date"], ascending=[True])


def merge_concat_df(adj_code, source_df, start_date, end_date, csv_file_path): #,
    """
    合并保存文件
    """
    datestart = datetime.datetime.strptime( start_date, '%Y-%m-%d' )
    dateend = datetime.datetime.strptime( end_date, '%Y-%m-%d' )
    #使用csv文件的最新日期加一天的数据，更新数据
    source_df = replace_fundamentals( adj_code, start_date, source_df )
    df_concat = source_df
    #对从csv最后一天加一天，到昨天的数据处理
    while datestart < dateend:
        date = datestart.strftime( '%Y-%m-%d' )
        price_df = get_price_df( adj_code, date, date )
        fundamentals_df = get_fundamentals_df( adj_code, date )
        if len( price_df ) > 0:
            df_merge = pd.merge( price_df, fundamentals_df, on="date", how="outer" )
            df_concat = pd.concat( [source_df, df_merge] ).sort_values(by=["date"], ascending=[True])
            source_df = df_concat
        datestart += datetime.timedelta( days=1 )

    source_df = df_concat
    print("更新前，最新的10条数据的日期是： %s" %(str(df_concat['date'].tolist()[-10:]) ))
    #获取当天的数据
    today_price_df = get_price_df( adj_code, end_date, end_date )
    today_fundamentals_df = get_fundamentals_df( adj_code, end_date )
    if len( today_price_df ) > 0:
        # 当天只能拿到昨天的数据，对cap字段单独处理
        today_fundamentals_df["cap"] = today_fundamentals_df.apply(
            lambda row: (row["capitalization"] * 10000 * today_price_df['close']) / today_price_df['factor'], axis=1 )
        #拿到的数据是昨天的，需要把日期换成今天的
        today_fundamentals_df["date"] = today_fundamentals_df.apply(lambda row: today_price_df['date'], axis=1)
        today_df_merge = pd.merge( today_price_df, today_fundamentals_df, on="date", how="outer" )
        df_concat = pd.concat( [source_df, today_df_merge] ).sort_values( by=["date"], ascending=[True] )
    print("更新后，最新的10条数据的日期是： %s" %(str(df_concat['date'].tolist()[-10:]) ))
    #保存
    df_concat.to_csv(csv_file_path, index=False)


def replace_fundamentals(adj_code, target_date, source_df):
    """
    把csv文件最新日期的财务数据与加一天的数据进行更新
    """
    fundamentals_df = get_fundamentals_df( adj_code, target_date )
    pe_ratio = fundamentals_df['pe_ratio'].values[-1]
    turnover_ratio = fundamentals_df['turnover_ratio'].values[-1]
    pb_ratio = fundamentals_df['pb_ratio'].values[-1]
    ps_ratio = fundamentals_df['ps_ratio'].values[-1]
    pcf_ratio = fundamentals_df['pcf_ratio'].values[-1]
    capitalization = fundamentals_df['capitalization'].values[-1]
    cap = fundamentals_df['cap'].values[-1]
    circulating_cap = fundamentals_df['circulating_cap'].values[-1]
    circulating_market_cap = fundamentals_df['circulating_market_cap'].values[-1]
    pe_ratio_lyr = fundamentals_df['pe_ratio_lyr'].values[-1]
    #替换
    source_df['pe_ratio'].values[-1] = pe_ratio
    source_df['turnover_ratio'].values[-1] = turnover_ratio
    source_df['pb_ratio'].values[-1] = pb_ratio
    source_df['ps_ratio'].values[-1] = ps_ratio
    source_df['pcf_ratio'].values[-1] = pcf_ratio
    source_df['capitalization'].values[-1] = capitalization
    source_df['cap'].values[-1] = cap
    source_df['circulating_cap'].values[-1] = circulating_cap
    source_df['circulating_market_cap'].values[-1] = circulating_market_cap
    source_df['pe_ratio_lyr'].values[-1] = pe_ratio_lyr
    return source_df


def _update(source_folder, target_date):
    count = 1
    for name in os.listdir( source_folder ):
        if (name.find(".csv") == -1 or name.find("_parsed.csv") > -1):
            # 如果不是原始数据就跳过该文件
            continue
        print("------------------------------------------------------------------------------------")
        print("第{}个文件".format(count), "，文件名是{}".format(name))
        code = name[:6]
        csv_file_path = os.path.join( source_folder, name )
        count += 1
        source_df = read_source_csv(csv_file_path)
        source_df = preprocess(source_df)
        #准备获取价格接口参数
        adj_code = get_adj_code(code)
        start_date, last_date = get_df_start_date(source_df)
        print("开始时间：{}".format(start_date))
        if last_date == target_date:
            print("原始数据的最近的日期就是今天，该文件不需要更新。")
        else:
            code_adj = get_adj_code(code)
            df_tmp = get_price( code_adj, start_date=last_date, end_date=target_date, frequency="daily",
                        fields=["close"], skip_paused=True )
            if len(df_tmp) <= 1:
                print("该股票还在停牌，没有数据需要更新。")
            else:
                #合并保存
                merge_concat_df(adj_code, source_df, start_date, target_date, csv_file_path) #
# -------------以上是与数据更新有关的代码---------


def update_data(data_folder, date=None):
    """
    更新指定目录里所有股票的csv文件。 
    目前csv文件为空也可以更新了。
    然后调用R脚本产生因子。
    :param data_folder: 这里面不仅有csv文件，还应该有R脚本，调用产生XXX_parsed.csv也在这个文件夹里。
    :param 更新数据 直到date 这一天（含）
    :return: 
    """
    if True:
        print("---------------------------开始更新原始数据------------------------------")
        # 最新下载数据的时间
        if date == None:
            target_date = datetime.datetime.now()
            target_date = target_date.strftime("%Y-%m-%d")
        else:
            target_date = date
        print("更新数据直到 %s 这一天（含）。" %(target_date))
        _update(source_folder=data_folder, target_date=target_date)
        print("---------------------------原始数据更新完毕------------------------------")

    if True:
        os.chdir(data_folder)
        print("原始数据更新完毕，接下来产生因子。")
        print()
        print("---------------------------以下为R脚本的输出-----------------------------")
        os.system("Rscript apply_getindicator.R")
        print()
        print("---------------------------以上为R脚本的输出-----------------------------")


def extract_data(data_folder, date_wanted, output_dir):
    """
    
    :param data_folder: 
    :param date_wanted: 
    :param output_dir: 输出到该文件夹下的 data.csv文件里
    :return: 
    """
    name = "data.csv"
    file_name = os.path.join(output_dir, name)
    count = 0
    count_halt = 0
    df_wanteddate = pd.DataFrame()
    for name in os.listdir(data_folder):
        if name.find("_parsed.csv") > -1:
            count += 1
            print(count, name)
            df_tmp = pd.read_csv(os.path.join(data_folder, name))
            df_tmp = preprocess(df_tmp)
            df_tmp = df_tmp[df_tmp["date"] == date_wanted]
            if len(df_tmp) == 0:
                print("%s %s 当天没数据。" % (date_wanted, name))
                count_halt += 1
            elif len(df_tmp) == 1:
                df_wanteddate = pd.concat([df_wanteddate, df_tmp])
            else:
                print("%s %s 数据有错误，请检查，退出。")
                sys.exit(-1)
    print("共有 %d 支股票，其中有 %d 支股票在 %s 没数据。" % (count, count_halt, date_wanted))
    df_wanteddate["id"] = range(1, len(df_wanteddate) + 1, 1)  # 创建一列从1开始的id列
    df_wanteddate.to_csv(os.path.join(output_dir, file_name), index=False)
    print("文件输出位置是：%s" %(os.path.join(output_dir, file_name)))


def generate_prediction(R_folder, output_folder, prediction_date=None):
    """
    
    :param R_folder: 目录的绝对路径，里面有Model 文件夹和Data_to_predict文件夹，对文件夹下的data.csv进行预测，
    输出也会存放在这个目录下的prediction.csv
    :param: output_folder: 输出目录的绝对路径，把生成的prediction.csv 拷贝到 此目录下的%Y-%d 文件夹下，并更名为prediction_%Y-%m-%d.csv
    :param: prediction_date: 此日期用于命名。
    :return: 
    """
    print("当前工作目录是 %s" %(os.getcwd()))
    print("切换工作目录至 %s" %(R_folder))
    os.chdir(R_folder)
    print()
    print("---------------------------以下为R脚本的输出-----------------------------")
    os.system("Rscript AtomPredict.R")
    print("---------------------------以上为R脚本的输出-----------------------------")
    print()
    print("模型选出的股票在 %s 下的 prediction.csv." %(R_folder))
    if prediction_date == None:
        prediction_date = datetime.datetime.now()
        prediction_date = prediction_date.strftime("%Y-%m-%d")
    datetime_format = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")
    year, week = datetime_format.isocalendar()[0], datetime_format.isocalendar()[1]  # 年份以及 是当年中的第几个星期
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("创建目录 %s." %(output_folder))
    tmp_dir = os.path.join(output_folder, "-".join([str(year), str(week)]))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        print("在 %s 下创建目录 %s." %(output_folder, "-".join([str(year), str(week)])))
    output_name = "prediction_" + prediction_date + ".csv"
    print("预测结果被拷贝到 %s." %(os.path.join(tmp_dir, output_name)))
    if os.path.exists(os.path.join(tmp_dir, output_name)):
        print("之前的文件将被覆盖。")
    copyfile(os.path.join(R_folder, "prediction.csv"), os.path.join(tmp_dir, output_name))


def generate_order(folder, cash, output_folder, date=None, cash_additional=5e4*0.2, top_n=10, rise_seuil=0.07):
    """
    
    :param folder: 该目录下必须有prediction.csv
    :param cash:
    :param top_n: 
    :return: 
    """
    name = "prediction.csv"
    df = pd.read_csv(os.path.join(folder, name))
    stock_list = df["tobuy"].values[0].split(",")
    target_date = datetime.datetime.now()
    target_date = target_date.strftime("%Y-%m-%d")
    # 获取最新的沪深300成分股名单
    curr_hs300 = get_index_stocks("000300.XSHG")
    # 只保留6位数代码，删除市场标识
    curr_hs300 = [name[0:6] for name in curr_hs300]
    stock_filtered = [name for name in stock_list if name in curr_hs300]
    stock_filtered_top = stock_filtered[:top_n]
    stock_list = stock_list[:top_n]
    print("选出来的前%d股票是：%s。" %(top_n, str(stock_list)))
    print("接下来剔除不合要求的股票以及计算买入份额。")
    if stock_filtered_top != stock_list:
        print("从中剔除下列已经不在沪深300成分股名单中的股票: %s，然后使用相邻排名的股票进行填充位置。" %(str(set(stock_list)-set(stock_filtered_top)) ) )
        print("最终选出来的前%d的股票是：%s。" %(top_n, stock_filtered_top))
        stock_list = stock_filtered_top.copy()
    else:
        print("选出的股票都在最新的沪深300成分股名单中，不需要剔除。")
    print("检查选出的股票里是否有ST标记的股票。")
    stock_to_remove = list()
    for stock in stock_list:
        df_tmp = get_extras("is_st", [get_adj_code(stock)], start_date=target_date, end_date=target_date)
        is_st = df_tmp.head(1)[df_tmp.columns[0]].values[0]
        if is_st:
            stock_to_remove.append(stock)
    if len(stock_to_remove) == 0:
        print("选出的股票中没有st股。")
    else:
        print("从选出的股票中剔除以下st股: %s" %(stock_to_remove))
        for stock in stock_to_remove:
            stock_list.remove(stock)

    def cal_amt(cash_per_stock, cash_additional, buy_price):
        """
        :param cash_per_stock: 每只股票分配的金额
        :param cash_additional: 额外资金
        :param buy_price: 买入价格
        :return: 
        """
        commission_buy_rate = 0.0003
        cash_to_use = cash_per_stock
        quantity = cash_to_use / ((1 + commission_buy_rate) * buy_price)
        quantity = floor(quantity / 100) * 100
        if quantity > 0:
            # 如果原始的钱购买至少100股，看看再加上额外资金后，能不能多买100股，如果能则多买100股
            cash_to_use = cash_per_stock + cash_additional
            quantity_tmp = cash_to_use / ((1 + commission_buy_rate) * buy_price)
            quantity_tmp = floor(quantity_tmp / 100) * 100
            if quantity_tmp > quantity:
                quantity += 100
            return quantity
        else:
            # 如果原始的钱不够买100股，那就再加上额外资金后，看看能不能买入100股
            cash_to_use = cash_per_stock + cash_additional
            quantity = cash_to_use / ((1 + commission_buy_rate) * buy_price)
            quantity = floor(quantity / 100) * 100
            return quantity

    cash_per_stock = cash / top_n
    print("可用现金的总额是：%f" %(cash))
    print("每支股票可用的现金是：%f" %(cash_per_stock))
    print("每支股票可用的额外浮动现金是：%f" %(cash_additional))
    entrust_price = list()
    entrust_quantity = list()
    for stock in stock_list:
        code_adj = get_adj_code(stock)
        price = get_price(code_adj, start_date=target_date, end_date=target_date, frequency="daily",
                        fields=["open", "high", "low", "close", "volume", "money", "factor", "pre_close"], skip_paused=True )
        if len(price) < 1:
            print("%s 在 %s 没有数据，停牌，不买入。" %(stock, target_date))
            entrust_price.append(0)
            entrust_quantity.append(0)
        else:
            curr_price = price["close"].values[0]
            pre_close = price["pre_close"].values[0]
            open_rise = 1.0*pre_close/curr_price - 1.0
            if open_rise > rise_seuil:
                print("%s 在 %s 开盘涨幅大于 %.2f%%，涨幅过大，不买入。" %(stock, target_date, 100*open_rise))
                entrust_price.append(0)
                entrust_quantity.append(0)
            else:
                entrust_price.append(curr_price)
                entrust_quantity.append(cal_amt(cash_per_stock, cash_additional, curr_price))
    df_order = pd.DataFrame(data=dict(stock=stock_list, price=entrust_price, quantity=entrust_quantity))
    df_order = df_order[df_order["price"] != 0]
    df_order["action"] = ["100 证券买入"] * len(df_order)
    # 调整顺序
    df_order = df_order[["stock", "action", "price", "quantity"]]
    df_order = df_order.rename(columns={"action": "买卖行为", "stock": "证券代码", "price": "委托价格", "quantity": "委托数量"})
    df_order.to_csv(os.path.join(folder, "order.txt"), index=False)

    if date == None:
        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d")
    datetime_format = datetime.datetime.strptime(date, "%Y-%m-%d")
    year, week = datetime_format.isocalendar()[0], datetime_format.isocalendar()[1]  # 年份以及 是当年中的第几个星期
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("创建目录 %s." %(output_folder))
    tmp_dir = os.path.join(output_folder, "-".join([str(year), str(week)]))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        print("在 %s 下创建目录 %s." %(output_folder, "-".join([str(year), str(week)])))
    output_name = "order_" + date + ".txt"
    print("订单被拷贝到 %s." %(os.path.join(tmp_dir, output_name)))
    if os.path.exists(os.path.join(tmp_dir, output_name)):
        print("之前的文件将被覆盖。")
    copyfile(os.path.join(folder, "order.txt"), os.path.join(tmp_dir, output_name))


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="脚本入口，可执行数据更新update，导出预测数据extract，获得预测股票predict, 生成订单order"
    )
    parser.add_argument(
        "--purpose",
        dest="purpose",
        help="执行目的，必须是以下中的一个，update, extract, predict, order"
    )
    parser.add_argument(
        "--data_folder",
        dest="data_folder",
        help="数据的存放路径，用于update，extract"
    )
    parser.add_argument(
        "--date_wanted",
        dest="date_wanted",
        help="导出哪一天的预测数据，用于extract"
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="预测数据data.csv 导出到哪个目录下，用于extract"
    )
    parser.add_argument(
        "--R_folder",
        dest="R_folder",
        help="R模型的存放目录，里面应该有Data_to_predict和Model两个文件夹，以及AtomPredict.R脚本，用于predict"
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        help="用于predict, order"
    )
    parser.add_argument(
        "--date",
        dest="date",
        help="%Y-%m-%d, 用于predict, order，也可用于update"
    )
    parser.add_argument(
        "--folder",
        dest="folder",
        help="订单的存放目录，用于order"
    )
    parser.add_argument(
        "--cash",
        dest="cash",
        help="用于购买股票的现金额，用于order"
    )
    args = parser.parse_args()
    purpose = args.purpose
    if purpose not in ["update", "extract", "predict", "order"]:
        print("执行目的，必须是以下选项其中之一，update, extract, predict, order %s是非法输入，退出。" %(purpose))
        sys.exit(-1)
    if purpose == "update":
        print("执行 update。")
        data_folder = args.data_folder
        date = args.date
        update_data(data_folder, date=date)
    elif purpose == "extract":
        print("执行 extract。")
        data_folder = args.data_folder
        date_wanted = args.date_wanted
        output_dir = args.output_dir
        extract_data(data_folder, date_wanted, output_dir)
    elif purpose == "predict":
        print("执行 predict。")
        R_folder = args.R_folder
        output_folder = args.output_folder
        prediction_date = args.date
        generate_prediction(R_folder, output_folder, prediction_date)
    elif purpose == "order":
        print("执行 order。")
        folder = args.folder
        cash = args.cash
        cash = float(cash)
        output_folder = args.output_folder
        date = args.date
        generate_order(folder, cash, output_folder, date)



# python script_entrance.py --purpose update --data_folder D:/data_shipan
# python script_entrance.py --purpose extract --data_folder D:/data_shipan --date_wanted 2018-06-14 --output_dir D:/wxd/Data_to_predict
# python script_entrance.py --purpose predict --R_folder D:/wxd --output_folder D:/wxd/result --date 2018-06-15
# python script_entrance.py --purpose order --folder D:/wxd --cash 1e5 --output_folder D:/wxd/result --date 2018-06-19
