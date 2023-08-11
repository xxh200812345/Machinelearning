from fuzzywuzzy import fuzz, process

from datetime import datetime

def get_month_number(abbreviation):
    """
    将3个字母的月份缩写转换为对应的月份数字
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    highest_similarity = 0
    most_similar_string = ""

    for month_str in months:
        month_str = month_str.upper()
        similarity = fuzz.hamming.normalized_similarity(abbreviation, month_str)
        print(f"{month_str} {similarity}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_string = month_str

    try:
        date_object = datetime.strptime(most_similar_string, "%b")
        month_number = date_object.month
        return month_number
    except ValueError:
        return f"月份{abbreviation}转换错误"
    
print(get_month_number('JUI'))