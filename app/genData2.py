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

