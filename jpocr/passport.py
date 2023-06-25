import os


# 护照对象
class Passport:
    def __init__(self, file_path):
        name, ext = os.path.splitext(os.path.basename(file_path))
        # 护照文件名
        self.file_name = name
        # PDF转PNG后保存文件名
        self.pdf2png_file_name = f"{name}_pdf2png.png"
        # 第一次裁剪后文件名
        self.cut_file_name = f"{name}_cut.png"
        # 图像优化后文件名
        self.edited_file_name = f"{name}_edited.png"
        # 签名文件名
        self.sign_file_name = f"{name}_sign.png"
        # 图像识别后文件
        self.tessract_file_name = f"{name}_tessract.png"
        # OCR数据
        self.data_list = []
        # 文件类型
        self.ext = ext
        # 护照信息
        self.info = {}
        self.info["err_msg"] = ""

    Type = "Type"
    Issuing_country = "Issuing country"
    Passport_No = "Passport No."
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

    # 图片存放位置
    image_dir = "images"

    PASSPORT_KEYS_LEN = {
        Type: 1,
        Issuing_country: 3,
        Passport_No: 9,
        Surname: 0,
        Given_name: 0,
        Nationality: 0,
        Date_of_birth: 9,
        Sex: 1,
        Registered_Domicile: 0,
        Date_of_issue: 9,
        Date_of_expiry: 9,
        foot1: 44,
        foot2: 44,
    }
    PASSPORT_KEYS_POSITION = {
        Type: (428.0, 67.0),
        Issuing_country: (640.0, 66.0),
        Passport_No: (981.0, 62.5),
        Surname: (480.5, 136.0),
        Given_name: (492.0, 205.5),
        Nationality: (480.5, 275.5),
        Date_of_birth: (770.0, 276.0),
        Sex: (426.5, 345.0),
        Registered_Domicile: (689.5, 345.0),
        Date_of_issue: (558.5, 415.5),
        Date_of_expiry: (557.5, 485.0),
        foot1: (624.5, 670.0),
        foot2: (624.5, 737.5),
    }

    PASSPORT_MRZ1_POSITION = {
        Type: (0, 1),
        Issuing_country: (2, 5),
        Surname: (5, -1),
        Given_name: (-1, -1),
    }
    PASSPORT_MRZ2_POSITION = {
        Passport_No: (0, 9),
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
