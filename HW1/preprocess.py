import os
import re

def DFS_file_search(dict_name):
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:  
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name)  
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)  
        except NotADirectoryError:
            result_txt.append(temp_name)
    return result_txt

path_list = DFS_file_search(r".\金庸小说")  # path_list 为包含所有小说文件的路径列表

#获得语料库
corpus = [] # corpus 存储语料库，其中以自然段为分割
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "").replace("本书来自www.cr173.com免费txt小说下载站", "").replace("更多更新免费电子书请关注www.cr173.com", "") for line in file]
        corpus += text

str = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  #过滤字符
for j in range(len(corpus)):
    corpus[j] = re.sub(str, "", corpus[j])

with open("preprocess.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        if len(line) > 1:
            print(line, file=f)     
