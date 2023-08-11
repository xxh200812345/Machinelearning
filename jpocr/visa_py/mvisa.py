import os
from mdocument import Document


# 护照对象
class Visa(Document):
    def __init__(self, file_path, index):
        super().__init__(file_path, index)

    CATEGORY = "CATEGORY"
    ENTRIES = "ENTRIES"
    ENTER_BEFORE = "ENTER BEFORE"
    DURATION_OF_EACH_STAY = "DURATION OF EACH STAY"
    ISSUE_DATE = "ISSUE DATE"
    FULL_NAME = "FULL NAME"
    BIRTH_DATE = "BIRTH DATE"
    PASSPORT_NO = "PASSPORT NO"
    foot1 = "foot1"
    foot2 = "foot2"

    OUT_ERROR_TAG = "Error"

    VISA_KEYS_LEN = {
        CATEGORY : 1,
        ENTRIES : 0,
        ENTER_BEFORE : 9,
        DURATION_OF_EACH_STAY : 3,
        ISSUE_DATE : 9,
        FULL_NAME : 0,
        BIRTH_DATE : 9,
        PASSPORT_NO : 9,
        foot1 : 0,
        foot2 : 0,
    }

    # x1,y1 x2,y2
    VISA_KEYS_POSITION = {
        CATEGORY : ((322,166),(348,203)),
        ENTRIES : ((1000,161),(1122,209)),
        ENTER_BEFORE : ((327,252),(559,295)),
        DURATION_OF_EACH_STAY : ((1001,253),(1082,292)),
        ISSUE_DATE : ((326,339),(558,382)),
        FULL_NAME : ((319,426),(663,463)),
        BIRTH_DATE : ((326,511),(558,550)),
        PASSPORT_NO : ((1000,514),(1238,551)),
        foot1 : ((106,787),(1515,821)),
        foot2 : ((106,863),(1515,900)),
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

    def get_nationality_code(nationality):
        country_codes = {
            "Japan": "JPN",
            "Albania": "ALB",
            "Algeria": "DZA",
            # 其他国家的映射关系...
        }
        return country_codes.get(
            nationality.capitalize(), "UNK"
        )  # 如果找不到对应的国家，则返回UNK（未知）
