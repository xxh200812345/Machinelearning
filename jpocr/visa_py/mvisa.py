import os
from mdocument import Document


# 护照对象
class Visa(Document):
    def __init__(self, file_path, index):
        super().__init__(file_path, index)

        self.type = None

    foot1 = "foot1"
    foot2 = "foot2"

    OUT_ERROR_TAG = "Error"

    TYPE_NAME_MID = 1 # 标签“中华人民共和国”在中间
    TYPE_NAME_LEFT = 2 # 标签“中华人民共和国签证”在左边

    def type1(self):
        self.CATEGORY = "CATEGORY"
        self.ENTRIES = "ENTRIES"
        self.ENTER_BEFORE = "ENTER BEFORE"
        self.DURATION_OF_EACH_STAY = "DURATION OF EACH STAY"
        self.ISSUE_DATE = "ISSUE DATE"
        self.FULL_NAME = "FULL NAME"
        self.BIRTH_DATE = "BIRTH DATE"
        self.PASSPORT_NO = "PASSPORT NO"

        self.VISA_KEYS_LEN = {
            self.CATEGORY : 1,
            self.ENTRIES : 0,
            self.ENTER_BEFORE : 9,
            self.DURATION_OF_EACH_STAY : 3,
            self.ISSUE_DATE : 9,
            self.FULL_NAME : 0,
            self.BIRTH_DATE : 9,
            self.PASSPORT_NO : 9,
            self.foot1 : 0,
            self.foot2 : 0,
        }

        # x1,y1 x2,y2
        self.VISA_KEYS_POSITION = {
            self.CATEGORY : ((322,166),(348,203)),
            self.ENTRIES : ((1000,161),(1122,209)),
            self.ENTER_BEFORE : ((327,252),(559,295)),
            self.DURATION_OF_EACH_STAY : ((1001,253),(1082,292)),
            self.ISSUE_DATE : ((326,339),(558,382)),
            self.FULL_NAME : ((319,426),(663,463)),
            self.BIRTH_DATE : ((326,511),(558,550)),
            self.PASSPORT_NO : ((1000,514),(1238,551)),
            self.foot1 : ((106,787),(1515,821)),
            self.foot2 : ((106,863),(1515,900)),
        }

        self.VISA_MRZ1_POSITION = {
            self.CATEGORY: (1, 2),
            self.FULL_NAME: (5, -1),
        }
        self.VISA_MRZ2_POSITION = {
            self.PASSPORT_NO: (0, 9),
            self.BIRTH_DATE: (13, 19),
            self.ENTER_BEFORE: (21, 27),
        }

    def type2(self):

        self.CATEGORY = "Category"
        self.ISSUE_DATE = "Issue Date"
        self.EXPIRY_DATE = "Expiry Date"
        self.ENTRIES = "Entries"
        self.DURATION_OF_STAY = "Duration of Stay"
        self.FULL_NAME = "Full Name"
        self.BIRTH_DATE = "Birth Date"
        self.PASSPORT_NO = "Passport Number"

        self.VISA_KEYS_LEN = {
            self.CATEGORY : 1,
            self.ENTRIES : 0,
            self.EXPIRY_DATE : 9,
            self.DURATION_OF_STAY : 3,
            self.ISSUE_DATE : 9,
            self.FULL_NAME : 0,
            self.BIRTH_DATE : 9,
            self.PASSPORT_NO : 9,
            self.foot1 : 0,
            self.foot2 : 0,
        }

        # x1,y1 x2,y2
        self.VISA_KEYS_POSITION = {
            self.CATEGORY : ((322,166),(348,203)),
            self.ENTRIES : ((1000,161),(1122,209)),
            self.EXPIRY_DATE : ((327,252),(559,295)),
            self.DURATION_OF_STAY : ((1001,253),(1082,292)),
            self.ISSUE_DATE : ((326,339),(558,382)),
            self.FULL_NAME : ((319,426),(663,463)),
            self.BIRTH_DATE : ((326,511),(558,550)),
            self.PASSPORT_NO : ((1000,514),(1238,551)),
            self.foot1 : ((106,787),(1515,821)),
            self.foot2 : ((106,863),(1515,900)),
        }

        self.VISA_MRZ1_POSITION = {
            self.CATEGORY: (1, 2),
            self.FULL_NAME: (5, -1),
        }
        self.VISA_MRZ2_POSITION = {
            self.PASSPORT_NO: (0, 9),
            self.BIRTH_DATE: (13, 19),
            self.EXPIRY_DATE: (21, 27),
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
