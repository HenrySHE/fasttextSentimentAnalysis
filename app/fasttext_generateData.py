#!/usr/bin/env python
# coding: utf-8

# ### 生成文本格式

# In[ ]:


import jieba
import pandas as pd
import random

# 设定各类类别映射，如'technology'为1，'car'为2……
cate_dic = {'car_positive':1, 'car_negative':2}
# 读取数据
df_car_positive = pd.read_csv("./t_processing_opinions_正面.csv", sep="\t", encoding='utf-8')
df_car_positive_text = df_car_positive['text'].dropna()
df_car_positive_text.drop_duplicates(keep='first',inplace=True)

df_car_negative = pd.read_csv("./t_processing_opinions_负面.csv", sep="\t", encoding='utf-8')
df_car_negative_text = df_car_negative['text'].dropna()
df_car_negative_text.drop_duplicates(keep='first',inplace=True)


# 转换为list列表的形式
# 正面dropna后有29995条, 去重后有26203条 → 训练数据10800条, 测试数据15403条
# 负面dropna后有22334条, 去重后有12003条 → 训练数据10800条, 测试数据1203条
print('开始打乱数据, 生成训练集和测试集')
car_positeive_list = df_car_positive_text.values.tolist()
car_negateive_list = df_car_negative_text.values.tolist()

random.shuffle(car_positeive_list)
random.shuffle(car_negateive_list)

car_positeive = car_positeive_list[:10800]
car_negateive = car_negateive_list[:10800]

car_positeive_test = car_positeive_list[10800:]
car_negateive_test = car_negateive_list[10800:]

# ### 载入停用词表 并定义本文处理函数，将文本处理为fasttext的输入格式

# In[ ]:


# stopwords=pd.read_csv("origin_data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# stopwords=stopwords['stopword'].values

stopwords = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            # 去标点、停用词等
            segs = list(filter(lambda x:len(x)>1, segs))
            segs = list(filter(lambda x:x not in stopwords, segs))
            # 将句子处理成  __label__1 词语 词语 词语 ……的形式
            sentences.append("__label__"+str(category)+" , "+" ".join(segs))
        except Exception as e:
            print(line)
            continue


# In[ ]:


#生成训练数据
sentences = []

preprocess_text(car_positeive, sentences, cate_dic['car_positive'])
preprocess_text(car_negateive, sentences, cate_dic['car_negative'])

sentences_test = []
preprocess_text(car_positeive_test, sentences_test, cate_dic['car_positive'])
preprocess_text(car_negateive_test, sentences_test, cate_dic['car_negative'])


# 随机打乱数据
random.shuffle(sentences)
random.shuffle(sentences_test)


# In[ ]:


# 将数据保存到train_data.txt中
print("writing data to fasttext format...")
out = open('train_data.txt', 'w', encoding='utf-8')
for sentence in sentences:
    out.write(sentence+"\n")
print("Finish processing training data !")
out.close()

out2 = open('test_data.txt', 'w', encoding='utf-8')
for sentence in sentences_test:
    out2.write(sentence+"\n")
print("Finish processing testing data !")
out2.close()


# ### 调用fastText训练生成模型

# In[ ]:


import fasttext
"""
  训练一个监督模型, 返回一个模型对象

  @param input:           训练数据文件路径
  @param lr:              学习率
  @param dim:             向量维度
  @param ws:              cbow模型时使用
  @param epoch:           次数
  @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
  @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
  @param minn:            构造subword时最小char个数
  @param maxn:            构造subword时最大char个数
  @param neg:             负采样
  @param wordNgrams:      n-gram个数
  @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
  @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
  @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
  @param lrUpdateRate:    学习率更新
  @param t:               负采样阈值
  @param label:           类别前缀
  @param verbose:         ??
  @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
  @return model object
"""
classifier = fasttext.train_supervised(input='train_data.txt', dim=100, epoch=5,
                                         lr=0.1, wordNgrams=2, loss='softmax')
classifier.save_model('classifier.model')


# ### 对模型效果进行评估

# In[ ]:


result = classifier.test('test_data.txt')
print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])




# In[ ]:
import fasttext

model = fasttext.load_model('classifier.model')

model.predict('')

outFile = open('test_data_neg.txt','a+',encoding='utf-8')

with open('tests_data.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if '__label__2' in line:
            outFile.write(line + '\n')

outFile.close()