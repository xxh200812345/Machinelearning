import re
import difflib

def check_similarity(string,pattern):
    
    similarity_threshold = 0.7

    similarity = difflib.SequenceMatcher(None, string, pattern).ratio()
    return similarity >= similarity_threshold

# 示例字符串
string = "PASsP0RT"
pattern = r"PASSPORT"

# 检查字符串与"jpn"的相似度
result = check_similarity(string,pattern)
print(result)
