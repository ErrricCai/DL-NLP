import jieba
import os
import re
import numpy as np
from gensim import corpora, models
from sklearn import cluster

# 数据预处理
def get_data(): # 如果文档还没分词，就进行分词
    outfilename_1 = "./train_jieba.txt"
    outfilename_2 = "./test_jieba.txt"
    if not os.path.exists('./train_jieba.txt'):
        outputs = open(outfilename_1, 'w', encoding='UTF-8')
        outputs_test = open(outfilename_2, 'w', encoding='UTF-8')
        datasets_root = "./dataset"
        catalog = "inf.txt"

        test_num = 10
        test_length = 20
        with open(os.path.join(datasets_root, catalog), "r", encoding='utf-8') as f:
            all_files = f.readline().split(",")
            print(all_files)

        for name in all_files:
            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='utf-8') as f:
                file_read = f.readlines()
                train_num = len(file_read) - test_num
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                train_text = ""
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                    outputs.write(line_seg.strip() + '\n')


                for test in choice_index[train_num:test_num + train_num]:
                    if test + test_length >= len(file_read):
                        continue
                    test_line = ""
                    line = file_read[test]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    for term in seg_list:
                        test_line += term + " "
                    outputs_test.write(test_line.strip()+'\n')

        outputs.close()
        outputs_test.close()
        print("得到训练集与测试集")

# 运用Word2Vec进行训练
if __name__ == "__main__":
    get_data() 
    fr = open('./train_jieba.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)

    num_features = 300  
    min_word_count = 10  
    num_workers = 16  
    context = 10  
    downsampling = 1e-3  
    sentences = models.word2vec.Text8Corpus("./train_jieba.txt")

    model = models.word2vec.Word2Vec(sentences, workers=num_workers,vector_size=num_features, min_count=min_word_count,window=context, sg=1, sample=downsampling)

# Kmeans 聚类
    names=[]
    for line in open("./金庸小说全人物.txt","r",encoding='utf-8'):
        line = line.strip('\n')
        names.append(line)
    names = [name for name in names if name in model.wv]
    name_vectors = [model.wv[name] for name in names]
    n=10
    label = cluster.KMeans(n).fit(name_vectors).labels_
    print(label)
    for i in range(n):
        print("\n类别" + str(i+1) + ":")
        for j in range(len(label)):
            if label[j] == i:
                print(names[j],end=" ")


    print('\n')
    test_name = ['乔峰', '杨过', '张无忌', '郭靖', '韦小宝']
    test_menpai = ['逍遥派', '全真教', '明教', '少林', '华山派']
    for i in range(5):
        print("与" + str(test_name[i]) + "关系相近的10个词汇:\n" + str(model.wv.most_similar(test_name[i], topn=10)))
    for i in range(5):
        print("与" + str(test_menpai[i]) + "关系相近的10个词汇:\n" + str(model.wv.most_similar(test_menpai[i], topn=10)))    

