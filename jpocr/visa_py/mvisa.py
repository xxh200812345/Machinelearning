import os
from mdocument import Document


# 护照对象
class Visa(Document):
    def __init__(self, file_path, index):
        super().__init__(file_path, index)

    Type = "Type"
    Issuing_country = "Issuing country"
    Visa_No = "Visa No."
    Surname = "Surname"
    Given_name = "Given name"
    Nationality = "Nationality"
    Date_of_birth = "Date of birth"
    Sex = "Sex"
    Registered_Domicile = "Registered Domicile"
    Date_of_issue = "Date of issue"
    Date_of_expiry = "Date of expiry"
    foot1 = "foot1"
    foot2 = "foot2"

    OUT_ERROR_TAG = "Error"

    VISA_KEYS_LEN = {
        Type: 1,
        Issuing_country: 3,
        Visa_No: 9,
        Surname: 0,
        Given_name: 0,
        Nationality: 0,
        Date_of_birth: 9,
        Sex: 1,
        Registered_Domicile: 0,
        Date_of_issue: 9,
        Date_of_expiry: 9,
        foot1: 0,
        foot2: 0,
    }
    VISA_KEYS_POSITION = {
        Type: ((423, 59), (442, 86)),
        Issuing_country: ((614, 58), (686, 86)),
        Visa_No: ((872, 52), (1103, 84)),
        Surname: ((425, 129), (497, 158)),
        Given_name: ((424, 200), (631, 229)),
        Nationality: ((425, 271), (551, 300)),
        Date_of_birth: ((642, 271), (929, 302)),
        Sex: ((423, 343), (442, 371)),
        Registered_Domicile: ((641, 342), (848, 371)),
        Date_of_issue: ((425, 413), (713, 445)),
        Date_of_expiry: ((425, 485), (713, 517)),
        foot1: ((45, 675), (1228, 705)),
        foot2: ((44, 743), (1228, 776)),
    }

    VISA_MRZ1_POSITION = {
        Type: (0, 1),
        Issuing_country: (2, 5),
        Surname: (5, -1),
        Given_name: (-1, -1),
    }
    VISA_MRZ2_POSITION = {
        Visa_No: (0, 9),
        Nationality: (10, 13),
        Date_of_birth: (13, 19),
        Sex: (20, 21),
        Date_of_expiry: (21, 27),
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
