#!/usr/bin/env python
# coding: utf-8

import fasttext


outFile = open('test_data_neg.txt','a+',encoding='utf-8')

with open('test_data.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if '__label__2' in line:
            outFile.write(line + '\n')

outFile.close()

