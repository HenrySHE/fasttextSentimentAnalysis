#!/usr/bin/env python
# coding: utf-8

# ### 生成文本格式

# In[ ]:


import jieba
import pandas as pd
import random

# 设定各类类别映射，如'technology'为1，'car'为2……
cate_dic = {'technology':1, 'car':2, 'entertainment':3, 'military':4, 'sports':5}
# 读取数据
df_technology = pd.read_csv("./origin_data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("./origin_data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./origin_data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./origin_data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("./origin_data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()
# 转换为list列表的形式
technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]


# ### 载入停用词表 并定义本文处理函数，将文本处理为fasttext的输入格式

# In[ ]:


stopwords=pd.read_csv("origin_data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

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

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

# 随机打乱数据
random.shuffle(sentences)


# In[ ]:


# 将数据保存到train_data.txt中
print("writing data to fasttext format...")
out = open('train_data.txt', 'w', encoding='utf-8')
for sentence in sentences:
    out.write(sentence+"\n")
print("done!")


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


result = classifier.test('train_data.txt')
print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])

