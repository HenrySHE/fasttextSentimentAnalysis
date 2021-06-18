# -*- coding: utf-8 -*-

from fastapi import FastAPI
import fasttext
import jieba


app = FastAPI(
    title="整体情感判定接口",
    description="训练fastAPI模型",
    version="1.0.0",
)


@app.post("/")
def root(content: str):
    cut_result = cut_sent(content)
    print('Cut result: %s'%cut_result)
    result = model.predict(cut_result)
    return {"result": str(result)}


# 命令行运行
# uvicorn main:app --reload


def load_model():
    model_instance = fasttext.load_model('classifier.model')
    return model_instance


def load_stopwords():
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def cut_sent(sentence: str):
    try:
        segs = jieba.lcut(sentence)
        segs = list(filter(lambda x: len(x) > 1, segs))
        segs = list(filter(lambda x: x not in stopwords, segs))
    except Exception as e:
        print('cut sent error')
        return sentence
    else:
        return str(" ".join(segs))


# 服务器运行的命令
if __name__ == "__main__":
    model = load_model()
    stopwords = load_stopwords()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9377)
