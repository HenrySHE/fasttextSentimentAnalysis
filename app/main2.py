# -*- coding: utf-8 -*-

from fastapi import FastAPI
import fasttext
import jieba
from pydantic import BaseModel


app = FastAPI(
    title="整体情感判定接口",
    description="训练fastAPI模型",
    version="1.0.0",
)


class PreLoad(BaseModel):
    model: None
    stopwords: list


def load_model():
    model_instance = fasttext.load_model('classifier.model')
    return model_instance


def load_stopwords():
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


@app.on_event("startup")
def startup_event():
    print('加载模型...')
    PreLoad.model = load_model()
    print('加载停用词...')
    PreLoad.stopwords = load_stopwords()
    print("加载停用词完成,长度是:%d "%(len(PreLoad.stopwords)))


class ContentInfo(BaseModel):
    content: str


@app.post("/")
def root(content: str):
    cut_result = cut_sent(content)
    print('Cut result: %s'%cut_result)
    result = PreLoad.model.predict(cut_result)
    return {"result": str(result)}


@app.post("/contentAnalysis")
def content_analysis(content_info: ContentInfo):
    try:
        (mark, prob) = PreLoad.model.predict(content_info.content)
    except Exception:
        print('Load Model Error')
        return {'mark': "Error", "prob": "-1"}
    else:
        if '__label__1' in str(mark):
            mark_result = "3"
        else:
            mark_result = "1"
        prob_result = prob[0]
        return {"mark": str(mark_result), "prob": str(prob_result)}


@app.post("/contentAnalysisWithSegment")
def content_analysis_with_segment(content_info: ContentInfo):
    processed_sent = cut_sent(content_info.content)
    try:
        (mark, prob) = PreLoad.model.predict(processed_sent)
    except Exception:
        print('Load Model Error')
        return {'mark': "Error", "prob": "-1"}
    else:
        if '__label__1' in str(mark):
            mark_result = "非负面"
        else:
            mark_result = "负面"
        prob_result = prob[0]
        return {"mark": str(mark_result), "prob": str(prob_result)}


# 命令行运行
# uvicorn main:app --reload


def cut_sent(sentence: str):
    try:
        segs = jieba.lcut(sentence)
        segs = list(filter(lambda x: len(x) > 1, segs))
        segs = list(filter(lambda x: x not in PreLoad.stopwords, segs))
    except Exception as e:
        print('cut sent error')
        return sentence
    else:
        return str(" ".join(segs))


# 服务器运行的命令
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9377)
