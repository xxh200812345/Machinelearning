import json

def set_to_tuple(data):
    if isinstance(data, set):
        # 如果是set对象，将其转换为列表并递归处理子元素
        return tuple(set_to_tuple(item) for item in data)
    elif isinstance(data, dict):
        # 如果是字典，将键和值都递归地转换为元组
        return tuple((set_to_tuple(key), set_to_tuple(value)) for key, value in data.items())
    elif isinstance(data, (list, tuple)):
        # 如果是列表或元组，递归处理每个元素
        return tuple(set_to_tuple(item) for item in data)
    else:
        # 其他类型的数据保持不变
        return data

def set_to_dict(data):
    if isinstance(data, set):
        # 如果是set对象，递归处理每个元素并转换为字典对象
        return {set_to_dict(item): None for item in data}
    elif isinstance(data, dict):
        # 如果是字典，递归处理键和值
        return {set_to_dict(key): set_to_dict(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        # 如果是列表或元组，递归处理每个元素
        return [set_to_dict(item) for item in data]
    else:
        # 其他类型的数据保持不变
        return data

# 示例包含多个嵌套set的数据结构
data = {'err_msg': '', 'main_info': {'Type': 'P', 'Issuing country': 'JPN', 'Passport No.': 'TR7261574', 'Surname': 'KINAMI', 'Given name': 'MAKOTO', 'Nationality': 'JAPAN', 'Date of birth': 
'17DEC1966', 'Sex': 'M', 'Registered Domicile': 'TOKYO', 'Date of issue': '27DEC2016', 'Date of expiry': '27DEC2026'}, 'mrz_info': {'Type': 'P', 'Issuing country': 'JPN', 'Surname': 'KINAMI', 'Given name': 'MAKOTO', 'Passport No.': 'TR7261574', 'Nationality': 'JPN', 'Date of birth': '661217', 'Sex': 'M', 'Date of expiry': '261227'}, 'vs_info': {'Type': '', 'Issuing country': '', 'Passport No.': '', 'Surname': '', 'Given name': '', 'Nationality': '', 'Date of birth': '', 'Sex': '', 'Registered Domicile': '', 'Date of issue': '', 'Date of expiry': 
'', 'foot1': '', 'foot2': ''}, 'file_name': {'MX-2630FN_20230714_152625_00'}, 'time': '2023-08-03 12:47:26', 'ocr_texts': 'P ((417, 56), (435, 83))\nJPN ((603, 54), (672, 82))\nTR7261574 ((856, 52), (1087, 83))\nKINAMI ((416, 124), (565, 152))\nMAKOTO ((416, 192), (567, 220))\nJAPAN ((418, 261), (541, 290))\n17DEC1966 ((628, 259), (909, 292))\nM ((416, 331), (434, 359))\nTOKYO ((627, 330), (750, 358))\n27DEC2016 ((416, 399), (698, 430))\n27DEC2026 ((416, 469), (698, 500))\nP<JPNKINAMI<<MAK0T0<<<<<<<<<<<<<<<<<<<<<<<<< ((44, 653), (1199, 684))\nTR72615744JPN6612175M2612270<<<<<<<<<<<<<<08 ((44, 716), (1199, 753))\n', 'foot1': 'P<JPNKINAMI<<MAK0T0<<<<<<<<<<<<<<<<<<<<<<<<<', 'foot2': 'TR72615744JPN6612175M2612270<<<<<<<<<<<<<<08', 'Type': 'P', 'Issuing country': 'JPN', 'Passport No.': 'TR7261574', 'Surname': 'KINAMI', 'Given name': 'MAKOTO', 'Nationality': 'JAPAN', 'Date of birth': '17DEC1966', 'Sex': 'M', 'Registered Domicile': 'TOKYO', 'Date of issue': '27DEC2016', 'Date of expiry': '27DEC2026'}

# 将数据结构递归地转换为元组
info = set_to_tuple(data)

path = r"H:\vswork\Machinelearning\jpocr\visa_py\output\jsons\a.json"
with open(path, "a", encoding="utf-8") as f:
    json_string = json.dump(set_to_dict(data), f, ensure_ascii=False)
