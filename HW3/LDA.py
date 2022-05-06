import jieba, os, re
import numpy as np
from gensim import corpora, models

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


if __name__ == "__main__":
    get_data() 
    fr = open('./train_jieba.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)


    # 构建词频矩阵，训练LDA模型
    dictionary = corpora.Dictionary(train)
    # corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
    # corpus是把每本小说ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=16)
    topic_list_lda = lda.print_topics(16)

    print("以LDA为分类器的16个主题的单词分布为：")
    for topic in topic_list_lda:
        print(topic)

    # 测试
    file_test = "./test_jieba.txt"
    news_test = open(file_test, 'r', encoding='UTF-8')
    test = []
    # 处理成正确的输入格式
    for line in news_test:
        line = [word.strip() for word in line.split(' ')]
        test.append(line)

    for text in test:
        corpus_test = dictionary.doc2bow((text))

    corpus_test = [dictionary.doc2bow(text) for text in test]
    
    # 得到每个测试集的主题分布
    topics_test = lda.get_document_topics(corpus_test)

    for i in range(10):
        print(f"{i}的主题分布为：{topics_test[i]}")

    fr.close()
    news_test.close()

