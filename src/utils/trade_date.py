# -*- coding: utf-8 -*-

"""
本脚本通过内置规则（硬编码节假日和调休日期）来计算A股交易日。
它不依赖任何外部API或网络请求。

核心逻辑:
1. 一个日期首先被判断是否为周末（周六/日）。
2. 如果是周末，再判断它是否被设定为调休工作日。
3. 如果是工作日（周一至周五），再判断它是否被设定为法定节假日。

注意:
此方法需要每年底根据国务院发布的下一年节假日安排来手动更新 HOLIDAY_DATA 字典。
"""

from datetime import date, timedelta, datetime

# --- 数据源: 中国国务院办公厅发布的节假日安排 ---
# 'holidays' (法定节假日): 在这些日期，即使是工作日也不开市。
# 'extra_workdays' (调休工作日): 在这些日期，即使是周末也要开市。
HOLIDAY_DATA = {
    2023: {
        'holidays': {
            # 元旦: 2022-12-31 to 2023-01-02, a total of 3 days. We only list days in 2023.
            date(2023, 1, 2),
            # 春节: Jan 21 to 27, a total of 7 days.
            date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26), date(2023, 1, 27),
            # 清明节: Apr 5, a total of 1 day.
            date(2023, 4, 5),
            # 劳动节: Apr 29 to May 3, a total of 5 days.
            date(2023, 5, 1), date(2023, 5, 2), date(2023, 5, 3),
            # 端午节: Jun 22 to 24, a total of 3 days.
            date(2023, 6, 22), date(2023, 6, 23),
            # 中秋国庆: Sep 29 to Oct 6, a total of 8 days.
            date(2023, 9, 29), date(2023, 10, 2), date(2023, 10, 3), date(2023, 10, 4), date(2023, 10, 5), date(2023, 10, 6),
        },
        'extra_workdays': {
            # 春节调休
            date(2023, 1, 28), date(2023, 1, 29),
            # 劳动节调休
            date(2023, 4, 23), date(2023, 5, 6),
            # 端午节调休
            date(2023, 6, 25),
            # 中秋国庆调休
            date(2023, 10, 7), date(2023, 10, 8),
        }
    },
    2024: {
        'holidays': {
            # 元旦
            date(2024, 1, 1),
            # 春节
            date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15), date(2024, 2, 16),
            # 清明节
            date(2024, 4, 4), date(2024, 4, 5),
            # 劳动节
            date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
            # 端午节
            date(2024, 6, 10),
            # 中秋节
            date(2024, 9, 16), date(2024, 9, 17),
            # 国庆节
            date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3), date(2024, 10, 4), date(2024, 10, 7),
        },
        'extra_workdays': {
            # 春节调休
            date(2024, 2, 4), date(2024, 2, 18),
            # 清明节调休
            date(2024, 4, 7),
            # 劳动节调休
            date(2024, 4, 28), date(2024, 5, 11),
            # 国庆节调休
            date(2024, 9, 29), date(2024, 10, 12),
        }
    },
    2025: {
        'holidays': {
            # 元旦
            date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3),
            # 春节
            date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 3),
            # 清明节
            date(2025, 4, 4),
            # 劳动节
            date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 5),
            # 端午节
            date(2025, 6, 2),
            # 中秋节
            date(2025, 9, 8),
            # 国庆节
            date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3), date(2025, 10, 6), date(2025, 10, 7),
        },
        'extra_workdays': {
            # 元旦调休
            date(2025, 1, 4),
            # 春节调休
            date(2025, 1, 26), date(2025, 2, 8),
            # 劳动节调休
            date(2025, 4, 27), date(2025, 5, 10),
            # 国庆节调休
            date(2025, 9, 28), date(2025, 10, 11),
        }
    }
}


"""
判断给定日期是否为A股交易日。

参数:
day (datetime.date): 需要判断的日期对象。

返回:
bool: 如果是交易日，返回 True；否则返回 False。
"""
def is_trading_day(day: date) -> bool:

    year = day.year
    if year not in HOLIDAY_DATA:
        # 对于没有数据的年份，只能进行简单的周末判断
        print(f"警告: 没有 {year} 年的节假日数据，将仅根据周末进行判断。")
        return day.weekday() < 5 # 0-4 represent Monday to Friday

    holidays = HOLIDAY_DATA[year]['holidays']
    extra_workdays = HOLIDAY_DATA[year]['extra_workdays']

    # 规则1: 如果日期在调休工作日列表中，则一定是交易日
    if day in extra_workdays:
        return True

    # 规则2: 如果日期在法定节假日列表中，则一定不是交易日
    if day in holidays:
        return False

    # 规则3: 如果是周六或周日，则不是交易日
    if day.weekday() >= 5:  # Saturday is 5, Sunday is 6
        return False

    # 规则4: 其他情况（正常工作日），是交易日
    return True


# 获取指定年份的所有A股交易日期。
# 参数:
#     year (int): 需要查询的年份 (例如: 2024)
# 返回:
#     list: 包含该年份所有交易日期字符串（格式 'YYYY-MM-DD'）的列表。

def get_trading_days_for_year(year: int) -> list:

    trading_days = []
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    current_day = start_date
    while current_day <= end_date:
        if is_trading_day(current_day):
            trading_days.append(current_day.strftime('%Y-%m-%d'))
        current_day += timedelta(days=1)
        
    return trading_days

def str_to_date(date_str: str) -> date | None:
    """Helper function to convert string to date object."""
    for fmt in ('%Y-%m-%d', '%Y%m%d'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    print(f"日期格式错误: '{date_str}'。请输入 'YYYY-MM-DD' 或 'YYYYMMDD' 格式。")
    return None

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 示例1: 获取2024年所有的交易日 ---
    print("\n--- 示例1: 获取2024年所有交易日 ---")
    trading_days_2024 = get_trading_days_for_year(2024)
    if trading_days_2024:
        print(f"根据内置规则，2024年共有 {len(trading_days_2024)} 个交易日。")
        print("前10个交易日是:", trading_days_2024[:10])
        print("最后10个交易日是:", trading_days_2024[-10:])

    # --- 示例2: 判断特定日期是否为交易日 (以2024年国庆节和调休为例) ---
    print("\n--- 示例2: 判断特定日期 ---")
    
    # 正常工作日
    date_to_check_1 = str_to_date("2024-09-27") # 周五, 交易日
    # 法定节假日
    date_to_check_2 = str_to_date("2024-10-01") # 周二, 国庆节, 非交易日
    # 正常周末
    date_to_check_3 = str_to_date("2024-09-28") # 周六, 非交易日
    # 因国庆调休，变为工作日的周日
    date_to_check_4 = str_to_date("2024-09-29") # 周日, 但调休上班, 交易日
    
    if date_to_check_1:
        print(f"'{date_to_check_1}' 是交易日吗? -> {is_trading_day(date_to_check_1)}")
    if date_to_check_2:
        print(f"'{date_to_check_2}' 是交易日吗? -> {is_trading_day(date_to_check_2)}")
    if date_to_check_3:
        print(f"'{date_to_check_3}' 是交易日吗? -> {is_trading_day(date_to_check_3)}")
    if date_to_check_4:
        print(f"'{date_to_check_4}' 是交易日吗? -> {is_trading_day(date_to_check_4)}")

    # --- 示例3: 判断今天是否为交易日 ---
    print("\n--- 示例3: 判断今天 ---")
    today = date.today()
    if is_trading_day(today):
        print(f"今天 ({today}) 是A股交易日。")
    else:
        print(f"今天 ({today}) 不是A股交易日。")

