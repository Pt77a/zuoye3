import os
import jieba
import logging
from gensim.models import word2vec


#文件夹路径
file_path = r'D:\Deeplearning'
#存放单独文件路径
filePaths=[]
#语料库序列
file_text = ""

for root, dirs, files in os.walk(file_path):
#遍历目录和其子目录，并返回一个三元组（root, dirs, files）的生成器
#root：目录路径。
#files：文件名列表。
    for name in files:
        if name.endswith(".txt"):  # 文件是否为txt文件
            filePath = os.path.join(root, name)#将文件名与路径相连
            filePaths.append(filePath)#存放所有文件的路径
            file1 = open(filePath, 'r', encoding='utf-8')#打开当前txt文件、只读
            text = file1.read()#读取当前文件的所有内容
            file1.close()#关闭文件
            file_text += text#将当前小说存入数据集中

file_text= file_text.replace("\n", "")
file_text = file_text.replace("\u3000", "")
file_text= file_text.replace(' ', "")
file_text = file_text.replace("本书来自www.cr173.com免费txt小说下载站", "")
file_text = file_text.replace("更多更新免费电子书请关注www.cr173.com", "")
file_text = file_text.replace("「", "")
file_text = file_text.replace("」", "")
file_text = file_text.replace("『", "")
file_text = file_text.replace("』", "")


words = []
words.append(file_text)
words = jieba.cut(file_text)
result = ' '.join(words)
with open(r'D:\分词.txt', 'w', encoding="utf-8") as f2:
    f2.write(result)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence(r'D:\分词.txt')

model = word2vec.Word2Vec(sentences=sentences, vector_size=200, min_count=10, window=5, sg=1, workers=4, epochs=50)

model.save(r'D:\word2vec.model')
