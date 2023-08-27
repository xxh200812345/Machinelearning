from mdocument import Document
from datetime import datetime
import re


TYPE_NAME_MID = 1  # 标签“中华人民共和国”在中间
TYPE_NAME_LEFT = 2  # 标签“中华人民共和国签证”在左边
OUT_ERROR_TAG = "Error"

I_MAIN_INFO = "main_info"
I_MRZ_INFO = "mrz_info"
I_VS_INFO = "vs_info"
I_ERR_MSG = "err_msg"
I_TIME = "time"
I_FILE_NAME = "file_name"
I_OCR_TEXTS = "ocr_texts"


# 护照对象
class VisaTitleLeft(Document):
    def __init__(self, document):
        super().__init__(document.file_path, document.index)

        self.type = None

    foot1 = "foot1"
    foot2 = "foot2"

    CATEGORY = "CATEGORY"
    ENTRIES = "ENTRIES"
    ENTER_BEFORE = "ENTER BEFORE"
    DURATION_OF_EACH_STAY = "DURATION OF EACH STAY"
    ISSUE_DATE = "ISSUE DATE"
    FULL_NAME = "FULL NAME"
    BIRTH_DATE = "BIRTH DATE"
    PASSPORT_NO = "PASSPORT NO"

    VISA_KEYS_LEN = {
        CATEGORY: 1,
        ENTRIES: 0,
        ENTER_BEFORE: 9,
        DURATION_OF_EACH_STAY: 3,
        ISSUE_DATE: 9,
        FULL_NAME: 0,
        BIRTH_DATE: 9,
        PASSPORT_NO: 9,
        foot1: 0,
        foot2: 0,
    }

    # x1,y1 x2,y2
    VISA_KEYS_POSITION = {
        CATEGORY: ((322, 166), (348, 203)),
        ENTRIES: ((1000, 161), (1122, 209)),
        ENTER_BEFORE: ((327, 252), (559, 295)),
        DURATION_OF_EACH_STAY: ((1001, 253), (1082, 292)),
        ISSUE_DATE: ((326, 339), (558, 382)),
        FULL_NAME: ((319, 426), (663, 463)),
        BIRTH_DATE: ((326, 511), (558, 550)),
        PASSPORT_NO: ((1000, 514), (1238, 551)),
        foot1: ((106, 787), (1515, 821)),
        foot2: ((106, 863), (1515, 900)),
    }

    VISA_MRZ1_POSITION = {
        CATEGORY: (1, 2),
        FULL_NAME: (5, -1),
    }
    VISA_MRZ2_POSITION = {
        PASSPORT_NO: (0, 9),
        BIRTH_DATE: (13, 19),
        ENTER_BEFORE: (21, 27),
    }

    def datalist2info(self, data_list, img):
        """
        获取护照信息
        """
        ret = self.info
        ret[I_MAIN_INFO] = {}
        ret[I_MRZ_INFO] = {}
        ret[I_VS_INFO] = {}

        height, width = img.shape[:2]

        ocr_texts = ""
        for data in data_list:
            # 文本
            ocr_texts += f"{data.content} {data.position}\n"

        ret[I_FILE_NAME] = self.file_name

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ret[I_TIME] = now_str

        data_list_len = len(data_list)
        data_list_len_default = 11
        if data_list_len != data_list_len_default:
            err_msg = f"识别后文字信息行数不为{data_list_len_default}，当前识别{data_list_len}。"
            ret[I_ERR_MSG] = add_error_to_info(ret[I_ERR_MSG], err_msg)

        ret[I_OCR_TEXTS] = ocr_texts

        error_vals = 0

        # 通过Passport No确认位移
        PassportNo_data = None
        for data in data_list:
            rect = data.position
            text = data.content

            pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
            match = re.match(pattern, text.strip().lower())
            if match:
                PassportNo_data = data
                break

        x_offset = 0
        y_offset = 0
        if PassportNo_data:
            ((x1, y1), (x2, y2)) = PassportNo_data.position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            ((px1, py1), (px2, py2)) = self.VISA_KEYS_POSITION[self.PASSPORT_NO]
            p_center_x = (px1 + px2) / 2
            p_center_y = (py1 + py2) / 2

            # 计算中心点之间的偏移量
            x_offset = int(center_x - p_center_x)
            y_offset = int(center_y - p_center_y)

        for data in data_list:
            text = data.content
            ((x1, y1), (x2, y2)) = data.position

            if PassportNo_data:
                # 移动矩形的坐标
                x1 = x1 - x_offset
                y1 = y1 - y_offset
                x2 = x2 - x_offset
                y2 = y2 - y_offset

            if len(text) <= 3:
                x1 -= 30
                x2 += 20

            y1 -= 10
            y2 += 10

            x1, y1, x2, y2 = rect_vs_box(((x1, y1), (x2, y2)), width, height)

            data.position = ((x1, y1), (x2, y2))

        # MRZ
        data_list_foots = [data for data in data_list if len(data.content) > 30]
        if len(data_list_foots) == 2:
            if data_list_foots[0].position[0][1] < data_list_foots[1].position[0][1]:
                ret[self.foot1] = data_list_foots[0].content
                ret[self.foot2] = data_list_foots[1].content
            else:
                ret[self.foot1] = data_list_foots[1].content
                ret[self.foot2] = data_list_foots[0].content

        else:
            for data in data_list_foots:
                text = data.content

                pattern = r"V.*[<くぐ]{1}.*[<くぐ]{1}.*"
                match = re.match(pattern, text.strip().upper())
                if match:
                    ret[self.foot1] = data.content

                pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
                match = re.match(pattern, text.strip().upper())
                if match:
                    ret[self.foot2] = data.content

            if self.foot1 not in ret or not ret[self.foot1]:
                ret[self.foot1] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

            if self.foot2 not in ret or not ret[self.foot2]:
                ret[self.foot2] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

        for key, value in self.VISA_KEYS_POSITION.items():
            if "foot" in key:
                continue

            # 统计所有覆盖率
            overlap_percentages = []
            for data in data_list:
                rect = data.position
                text = data.content

                overlap_percentage = get_overlap_percentage(value, rect)
                if overlap_percentage != 0:
                    overlap_percentages.append((overlap_percentage, text))

            # 找到覆盖率最大的值
            if len(overlap_percentages) != 0:
                max_overlap_percentage = 0
                max_overlap_text = ""
                for overlap_percentage, text in overlap_percentages:
                    if overlap_percentage > max_overlap_percentage:
                        max_overlap_percentage = overlap_percentage
                        max_overlap_text = text
                ret[key] = max_overlap_text

            if key not in ret:
                ret[key] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

        if error_vals > 0:
            err_msg = f"一共有{error_vals}个数据没有找到对应值。"
            ret[I_ERR_MSG] = add_error_to_info(ret[I_ERR_MSG], err_msg)

        # 根据基础信息生成三个对象，对象main_info保存护照主要信息，对象mrz_info保存下方mrz分解后的信息，对象vs_info保存对比信息
        check_len(ret, self)
        self.set_main_info(ret)
        self.set_mrz_info(ret)
        self.set_vs_info(ret)

        self.info = ret

        return data_list

    def set_main_info(self, ret):
        main_info = ret[I_MAIN_INFO]
        vs_info = ret[I_VS_INFO]

        main_info[self.CATEGORY] = ret[self.CATEGORY]
        main_info[self.ENTER_BEFORE] = to_O(ret[self.ENTER_BEFORE])

        tmp = self.ENTER_BEFORE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            main_info[tmp] = (
                to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
            )
        else:
            main_info[tmp] = ""

        replace_chat = ["Q", "O", "o"]
        temp_str = ret[self.DURATION_OF_EACH_STAY]
        for item in replace_chat:
            temp_str = temp_str.replace(item, "0")
        main_info[self.DURATION_OF_EACH_STAY] = temp_str

        tmp = self.ISSUE_DATE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            main_info[tmp] = (
                to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
            )
        else:
            main_info[tmp] = ""

        main_info[self.FULL_NAME] = to_O(ret[self.FULL_NAME])

        tmp = self.BIRTH_DATE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            main_info[tmp] = (
                to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
            )
        else:
            main_info[tmp] = ""

        tmp = self.PASSPORT_NO
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            main_info[tmp] = to_O(ret[tmp][:2]) + to_0(ret[tmp][2:])
        else:
            main_info[tmp] = ""

    def set_mrz_info(self, ret):
        mrz_info = ret[I_MRZ_INFO]
        vs_info = ret[I_VS_INFO]
        foot1 = to_O(ret[self.foot1])
        foot2 = ret[self.foot2]

        if vs_info[self.foot1][:5] != OUT_ERROR_TAG:
            for title, position in self.VISA_MRZ1_POSITION.items():
                if title == self.FULL_NAME:
                    find_surname = foot1[position[0] :]
                    split_array = find_surname.split("<")
                    cleaned_array = [item for item in split_array if item]

                    if len(cleaned_array) >= 2:
                        mrz_info[title] = f"{cleaned_array[1]}{cleaned_array[0]}"
                    else:
                        mrz_info[title] = OUT_ERROR_TAG + ": 没有找到姓名"

                if title == self.CATEGORY:
                    mrz_info[title] = foot1[position[0] : position[1]]
        else:
            for title, position in self.VISA_MRZ1_POSITION.items():
                mrz_info[title] = ""

        if vs_info[self.foot2][:5] != OUT_ERROR_TAG:
            for title, position in self.VISA_MRZ2_POSITION.items():
                tmp = foot2[position[0] : position[1]]

                if title == self.PASSPORT_NO:
                    mrz_info[title] = to_O(tmp[:2]) + to_0(tmp[2:])

                if title == self.BIRTH_DATE:
                    mrz_info[title] = to_0(tmp)

                if title == self.ENTER_BEFORE:
                    mrz_info[title] = to_0(tmp)

        else:
            for title, position in self.VISA_MRZ1_POSITION.items():
                mrz_info[title] = ""

    def set_vs_info(self, ret):
        main_info = ret[I_MAIN_INFO]
        mrz_info = ret[I_MRZ_INFO]
        vs_info = ret[I_VS_INFO]

        for title, mrz_item in mrz_info.items():
            if title in main_info == False:
                main_info[title] = f"main_info中不存在这个属性"
                continue

            if main_info[title][:5] == OUT_ERROR_TAG:
                error_msg = f"中间的信息项目存在错误"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif mrz_info[title][:5] == OUT_ERROR_TAG:
                error_msg = f"MRZ的信息项目存在错误"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif main_info == "":
                error_msg = f"中间的信息项目没有值"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif mrz_info == "":
                error_msg = f"MRZ的信息项目没有值"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif title == self.ENTER_BEFORE or title == self.BIRTH_DATE:
                month_num = get_month_number(main_info[title][2:5])
                if isinstance(month_num, str):
                    error_msg = month_num
                    vs_info[title] = add_error_to_info(vs_info[title], error_msg)
                else:
                    main_item = (
                        main_info[title][-2:]
                        + str(month_num).zfill(2)
                        + main_info[title][:2]
                    )
                    if main_item != mrz_item:
                        error_msg = f"数据不一致"
                        vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif main_info[title] != mrz_info[title]:
                error_msg = f"数据不一致"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        return


# 护照对象
class VisaTitleMID(Document):
    def __init__(self, document):
        super().__init__(document.file_path, document.index)

        self.type = None

    foot1 = "foot1"
    foot2 = "foot2"

    CATEGORY = "Category"
    ISSUE_DATE = "Issue Date"
    EXPIRY_DATE = "Expiry Date"
    ENTRIES = "Entries"
    DURATION_OF_STAY = "Duration of Stay"
    FULL_NAME = "Full Name"
    BIRTH_DATE = "Birth Date"
    PASSPORT_NO = "Passport Number"

    VISA_KEYS_LEN = {
        CATEGORY: 1,
        ENTRIES: 0,
        EXPIRY_DATE: 18,
        DURATION_OF_STAY: 3,
        ISSUE_DATE: 18,
        FULL_NAME: 0,
        BIRTH_DATE: 18,
        PASSPORT_NO: 18,
        foot1: 0,
        foot2: 0,
    }

    # x1,y1 x2,y2
    VISA_KEYS_POSITION = {
        CATEGORY: ((519, 128), (541, 165)),
        ISSUE_DATE: ((524, 215), (750, 259)),
        EXPIRY_DATE: ((812, 215), (1051, 259)),
        ENTRIES: ((519, 307), (564, 343)),
        DURATION_OF_STAY: ((812, 305), (889, 342)),
        FULL_NAME: ((518, 390), (972, 430)),
        BIRTH_DATE: ((758, 475), (1000, 516)),
        PASSPORT_NO: ((1000, 475), (1289, 516)),
        foot1: ((104, 776), (1634, 819)),
        foot2: ((105, 854), (1635, 900)),
    }

    VISA_MRZ1_POSITION = {
        CATEGORY: (1, 2),
        FULL_NAME: (5, -1),
    }
    VISA_MRZ2_POSITION = {
        PASSPORT_NO: (0, 9),
        BIRTH_DATE: (13, 19),
        EXPIRY_DATE: (21, 27),
    }

    def datalist2info(self, data_list, img):
        """
        获取护照信息
        """
        ret = self.info
        ret[I_MAIN_INFO] = {}
        ret[I_MRZ_INFO] = {}
        ret[I_VS_INFO] = {}

        ocr_texts = ""
        for data in data_list:
            # 文本
            ocr_texts += f"{data.content} {data.position}\n"

        ret[I_FILE_NAME] = self.file_name

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ret[I_TIME] = now_str

        data_list_len = len(data_list)
        data_list_len_default = 11
        if data_list_len != data_list_len_default:
            err_msg = f"识别后文字信息行数不为{data_list_len_default}，当前识别{data_list_len}。"
            ret[I_ERR_MSG] = add_error_to_info(ret[I_ERR_MSG], err_msg)

        ret[I_OCR_TEXTS] = ocr_texts

        error_vals = 0

        # MRZ
        data_list_foots = [data for data in data_list if len(data.content) > 30]
        if len(data_list_foots) == 2:
            if data_list_foots[0].position[0][1] < data_list_foots[1].position[0][1]:
                ret[self.foot1] = data_list_foots[0].content
                ret[self.foot2] = data_list_foots[1].content
            else:
                ret[self.foot1] = data_list_foots[1].content
                ret[self.foot2] = data_list_foots[0].content

        else:
            for data in data_list_foots:
                text = data.content

                pattern = r"V.*[<くぐ]{1}.*[<くぐ]{1}.*"
                match = re.match(pattern, text.strip().upper())
                if match:
                    ret[self.foot1] = data.content

                pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
                match = re.match(pattern, text.strip().upper())
                if match:
                    ret[self.foot2] = data.content

            if self.foot1 not in ret or not ret[self.foot1]:
                ret[self.foot1] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

            if self.foot2 not in ret or not ret[self.foot2]:
                ret[self.foot2] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

        for key, value in self.VISA_KEYS_POSITION.items():
            if "foot" in key:
                continue

            # 统计所有覆盖率
            overlap_percentages = []
            for data in data_list:
                rect = data.position
                text = data.content

                overlap_percentage = get_overlap_percentage(value, rect)
                if overlap_percentage != 0:
                    overlap_percentages.append((overlap_percentage, text))

            # 找到覆盖率最大的值
            if len(overlap_percentages) != 0:
                max_overlap_percentage = 0
                max_overlap_text = ""
                for overlap_percentage, text in overlap_percentages:
                    if overlap_percentage > max_overlap_percentage:
                        max_overlap_percentage = overlap_percentage
                        max_overlap_text = text
                ret[key] = max_overlap_text

            if key not in ret:
                ret[key] = OUT_ERROR_TAG + ": 没有找到数据."
                error_vals += 1

        if error_vals > 0:
            err_msg = f"一共有{error_vals}个数据没有找到对应值。"
            ret[I_ERR_MSG] = add_error_to_info(ret[I_ERR_MSG], err_msg)

        # 根据基础信息生成三个对象，对象main_info保存护照主要信息，对象mrz_info保存下方mrz分解后的信息，对象vs_info保存对比信息
        check_len(ret, self)
        self.set_main_info(ret)
        self.set_mrz_info(ret)
        self.set_vs_info(ret)

        self.info = ret

        return data_list

    def set_main_info(self, ret):
        main_info = ret[I_MAIN_INFO]
        vs_info = ret[I_VS_INFO]

        main_info[self.CATEGORY] = ret[self.CATEGORY]

        tmp = self.ISSUE_DATE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            tmp_val = ret[tmp][:9]
            main_info[tmp] = to_0(tmp_val[:2]) + to_O(tmp_val[2:5]) + to_0(tmp_val[-4:])
        else:
            main_info[tmp] = ""

        tmp = self.EXPIRY_DATE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            tmp_val = ret[tmp][-9:]
            main_info[tmp] = to_0(tmp_val[:2]) + to_O(tmp_val[2:5]) + to_0(tmp_val[-4:])
        else:
            main_info[tmp] = ""

        main_info[self.ENTRIES] = to_0(ret[self.ENTRIES])
        main_info[self.DURATION_OF_STAY] = to_0(ret[self.DURATION_OF_STAY])

        main_info[self.FULL_NAME] = to_O(ret[self.FULL_NAME])

        tmp = self.BIRTH_DATE
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            tmp_val = ret[tmp][:9]
            main_info[tmp] = to_0(tmp_val[:2]) + to_O(tmp_val[2:5]) + to_0(tmp_val[-4:])
        else:
            main_info[tmp] = ""

        tmp = self.PASSPORT_NO
        if vs_info[tmp][:5] != OUT_ERROR_TAG:
            tmp_val = ret[tmp][-9:]
            main_info[tmp] = to_O(tmp_val[:2]) + to_0(tmp_val[2:])
        else:
            main_info[tmp] = ""

    def set_mrz_info(self, ret):
        mrz_info = ret[I_MRZ_INFO]
        vs_info = ret[I_VS_INFO]
        foot1 = to_O(ret[self.foot1])
        foot2 = ret[self.foot2]

        if vs_info[self.foot1][:5] != OUT_ERROR_TAG:
            for title, position in self.VISA_MRZ1_POSITION.items():
                if title == self.FULL_NAME:
                    find_surname = foot1[position[0] :]
                    split_array = find_surname.split("<")
                    cleaned_array = [item for item in split_array if item]

                    if len(cleaned_array) >= 2:
                        mrz_info[title] = f"{cleaned_array[1]}{cleaned_array[0]}"
                    else:
                        mrz_info[title] = OUT_ERROR_TAG + ": 没有找到姓名"

                if title == self.CATEGORY:
                    mrz_info[title] = foot1[position[0] : position[1]]
        else:
            for title, position in self.VISA_MRZ1_POSITION.items():
                mrz_info[title] = ""

        if vs_info[self.foot2][:5] != OUT_ERROR_TAG:
            for title, position in self.VISA_MRZ2_POSITION.items():
                tmp = foot2[position[0] : position[1]]

                if title == self.PASSPORT_NO:
                    mrz_info[title] = to_O(tmp[:2]) + to_0(tmp[2:])

                if title == self.BIRTH_DATE:
                    mrz_info[title] = to_0(tmp)

                if title == self.EXPIRY_DATE:
                    mrz_info[title] = to_0(tmp)

        else:
            for title, position in self.VISA_MRZ1_POSITION.items():
                mrz_info[title] = ""

    def set_vs_info(self, ret):
        main_info = ret[I_MAIN_INFO]
        mrz_info = ret[I_MRZ_INFO]
        vs_info = ret[I_VS_INFO]

        for title, mrz_item in mrz_info.items():
            if title in main_info == False:
                main_info[title] = f"main_info中不存在这个属性"
                continue

            if main_info[title][:5] == OUT_ERROR_TAG:
                error_msg = f"中间的信息项目存在错误"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif mrz_info[title][:5] == OUT_ERROR_TAG:
                error_msg = f"MRZ的信息项目存在错误"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif main_info == "":
                error_msg = f"中间的信息项目没有值"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif mrz_info == "":
                error_msg = f"MRZ的信息项目没有值"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif (
                title == self.ISSUE_DATE
                or title == self.EXPIRY_DATE
                or title == self.BIRTH_DATE
            ):
                month_num = get_month_number(main_info[title][2:5])
                if isinstance(month_num, str):
                    error_msg = month_num
                    vs_info[title] = add_error_to_info(vs_info[title], error_msg)
                else:
                    main_item = (
                        main_info[title][-2:]
                        + str(month_num).zfill(2)
                        + main_info[title][:2]
                    )
                    if main_item != mrz_item:
                        error_msg = f"数据不一致"
                        vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif main_info[title] != mrz_info[title]:
                error_msg = f"数据不一致"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        return


def get_nationality_code(nationality):
    country_codes = {
        "Japan": "JPN",
        "Albania": "ALB",
        "Algeria": "DZA",
        # 其他国家的映射关系...
    }
    return country_codes.get(nationality.capitalize(), "UNK")  # 如果找不到对应的国家，则返回UNK（未知）


def setType(visa):
    if visa.type == TYPE_NAME_LEFT:
        return VisaTitleLeft(visa)
    if visa.type == TYPE_NAME_MID:
        return VisaTitleMID(visa)


def get_overlap_percentage(normal_rect, ocr_data_rect):
    """
    判断orc数据是否在标准矩形区域内

    参数:
    - normal_rect: 一个包含四个元素的元组或列表，表示点的坐标 (x1, y1, x2, y2)
    - ocr_data_rect: 一个包含四个元素的元组或列表，表示矩形的坐标 (x1, y1, x2, y2)

    返回值:
    - 如果点在矩形区域内，则返回 True，否则返回 False
    """
    normal_area = (normal_rect[1][0] - normal_rect[0][0]) * (
        normal_rect[1][1] - normal_rect[0][1]
    )  # 计算正常矩形的面积

    intersection_x = max(
        0,
        min(normal_rect[1][0], ocr_data_rect[1][0])
        - max(normal_rect[0][0], ocr_data_rect[0][0]),
    )  # 计算交集的宽度
    intersection_y = max(
        0,
        min(normal_rect[1][1], ocr_data_rect[1][1])
        - max(normal_rect[0][1], ocr_data_rect[0][1]),
    )  # 计算交集的高度
    intersection_area = intersection_x * intersection_y  # 计算交集的面积

    overlap_percentage = (intersection_area / normal_area) * 100  # 计算重叠的百分比

    if overlap_percentage > 0.1:
        return overlap_percentage
    else:
        return 0


def rect_vs_box(rect, width, height):
    """
    矩形(x1,y1,x2,y2)数据不能超过设置范围
    """

    ((x1, y1), (x2, y2)) = rect

    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = width if x2 > width else x2
    y2 = height if y2 > height else y2

    return (x1, y1, x2, y2)


def add_error_to_info(info, error_msg):
    """
    追加对应的比较错误 info：vs_info[title]
    """
    if info[:5] != OUT_ERROR_TAG:
        info = OUT_ERROR_TAG + ": " + error_msg
    else:
        info += ";" + error_msg

    return info


def check_len(ret, visa):
    for title, key_len in visa.VISA_KEYS_LEN.items():
        if key_len > 0 and len(ret[title]) != key_len:
            ret[I_VS_INFO][
                title
            ] = f"{OUT_ERROR_TAG}: 实际长度{len(ret[title])}不等于预测长度{key_len}"
        else:
            ret[I_VS_INFO][title] = ""


def to_O(text):
    text = text.replace("0", "O")
    text = text.replace("1", "I")
    return text


def to_0(text):
    replace_chat = ["Q", "O", "o"]
    for item in replace_chat:
        text = text.replace(item, "0")

    replace_chat = ["I", "L"]
    for item in replace_chat:
        text = text.replace(item, "1")

    return text


def get_month_number(abbreviation):
    """
    将3个字母的月份缩写转换为对应的月份数字
    """
    if abbreviation == "JUI":
        abbreviation = "JUL"
    try:
        date_object = datetime.strptime(abbreviation, "%b")
        month_number = date_object.month
        return month_number
    except ValueError:
        return f"月份{abbreviation}转换错误"