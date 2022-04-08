import jieba
from collections import Counter
import math
import time

with open("preprocess.txt", "r", encoding="utf-8") as f:
    corpus = [eve.strip("\n") for eve in f]

def zi(v):
    l=[]
    for eve in v:
        l.append(eve)
    return l

# 1-gram————————————————————————————————————————————————————————————————————————————————————————————————————————

# 基于词
before = time.time()
token_ci = []
for para in corpus:
    token_ci += jieba.lcut(para)
token_num_ci = len(token_ci)
ct1_ci = Counter(token_ci)
vocab1_ci = ct1_ci.most_common()
entropy_unigram_ci = sum([-(eve[1]/token_num_ci)*math.log((eve[1]/token_num_ci),2) for eve in vocab1_ci])
print("语料库总词数：", token_num_ci)
print("基于中文词的一元模型信息熵为:", round(entropy_unigram_ci,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")

#基于字
before = time.time()
token_zi = []
for para in corpus:
    token_zi += zi(para)
token_num_zi = len(token_zi)
ct1_zi = Counter(token_zi)
vocab1_zi = ct1_zi.most_common()
entropy_unigram_zi = sum([-(eve[1]/token_num_zi)*math.log((eve[1]/token_num_zi),2) for eve in vocab1_zi])
print("语料库总字数：", token_num_zi)
print("基于中文字的一元模型信息熵为:", round(entropy_unigram_zi,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")

# 2-gram————————————————————————————————————————————————————————————————————————————————————————————————————————

#基于词
before = time.time()
def combine2gram(cutword_list):
    if len(cutword_list) == 1:
        return []
    res = []
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i] + "s"+ cutword_list[i+1]) 
    return res

token_bigram_ci = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_bigram_ci += combine2gram(cutword_list)

# 2-gram的频率统计
token_bigram_num_ci = len(token_bigram_ci)
ct2_ci = Counter(token_bigram_ci)
vocab2_ci = ct2_ci.most_common()

# 2-gram相同句首的频率统计
same_1st_word_ci = [eve.split("s")[0] for eve in token_bigram_ci]
assert token_bigram_num_ci == len(same_1st_word_ci)
ct_1st_ci = Counter(same_1st_word_ci)
vocab_1st_ci = dict(ct_1st_ci.most_common())
entropy_bigram_ci = 0
for eve in vocab2_ci:
    p_xy = eve[1]/token_bigram_num_ci
    first_word = eve[0].split("s")[0]
    entropy_bigram_ci += -p_xy*math.log(eve[1]/vocab_1st_ci[first_word], 2)
print("语料库总词数：", token_bigram_num_ci)
print("基于中文词的二元模型信息熵为:", round(entropy_bigram_ci,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")

#基于字
before = time.time()
token_bigram_zi = []
for para in corpus:
    cutword_list = zi(para)
    token_bigram_zi += combine2gram(cutword_list)

token_bigram_num_zi = len(token_bigram_zi)
ct2_zi = Counter(token_bigram_zi)
vocab2_zi = ct2_zi.most_common()
same_1st_word_zi = [eve.split("s")[0] for eve in token_bigram_zi]
assert token_bigram_num_zi == len(same_1st_word_zi)
ct_1st_zi = Counter(same_1st_word_zi)
vocab_1st_zi = dict(ct_1st_zi.most_common())
entropy_bigram_zi = 0
for eve in vocab2_zi:
    p_xy = eve[1]/token_bigram_num_zi
    first_word = eve[0].split("s")[0]
    entropy_bigram_zi += -p_xy*math.log(eve[1]/vocab_1st_zi[first_word], 2)
print("语料库总字数：", token_bigram_num_zi)
print("基于中文字的二元模型信息熵为:", round(entropy_bigram_zi,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")

# 3-gram————————————————————————————————————————————————————————————————————————————————————————————————————————

#基于词
before = time.time()
def combine3gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        res.append(cutword_list[i] + cutword_list[i+1] + "s" + cutword_list[i+2] ) 
    return res

token_trigram_ci = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_trigram_ci += combine3gram(cutword_list)

# 3-gram的频率统计
token_trigram_num_ci = len(token_trigram_ci)
ct3_ci = Counter(token_trigram_ci)
vocab3_ci = ct3_ci.most_common()

# 3-gram相同句首两个词语的频率统计
same_2st_word_ci = [eve.split("s")[0] for eve in token_trigram_ci]
assert token_trigram_num_ci == len(same_2st_word_ci)
ct_2st_ci = Counter(same_2st_word_ci)
vocab_2st_ci = dict(ct_2st_ci.most_common())
entropy_trigram_ci = 0
for eve in vocab3_ci:
    p_xyz = eve[1]/token_trigram_num_ci
    first_2word = eve[0].split("s")[0]
    entropy_trigram_ci += -p_xyz*math.log(eve[1]/vocab_2st_ci[first_2word], 2)
print("语料库总词数：", token_trigram_num_ci)
print("基于中文词的三元模型信息熵为:", round(entropy_trigram_ci,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")

#基于字
before = time.time()
token_trigram_zi = []
for para in corpus:
    cutword_list = zi(para)
    token_trigram_zi += combine3gram(cutword_list)

token_trigram_num_zi = len(token_trigram_zi)
ct3_zi = Counter(token_trigram_zi)
vocab3_zi = ct3_zi.most_common()
same_2st_word_zi = [eve.split("s")[0] for eve in token_trigram_zi]
assert token_trigram_num_zi == len(same_2st_word_zi)
ct_2st_zi = Counter(same_2st_word_zi)
vocab_2st_zi = dict(ct_2st_zi.most_common())
entropy_trigram_zi = 0
for eve in vocab3_zi:
    p_xyz = eve[1]/token_trigram_num_zi
    first_2word = eve[0].split("s")[0]
    entropy_trigram_zi += -p_xyz*math.log(eve[1]/vocab_2st_zi[first_2word], 2)
print("语料库总字数：", token_trigram_num_zi)
print("基于中文字的三元模型信息熵为:", round(entropy_trigram_zi,5),"比特/词")
after = time.time()
print("运行时间：", round(after-before,5), "秒")