from gensim.models import Word2Vec

model = Word2Vec.load(r'D:\word2vec.model')#加载模型，

'''查询词之间的相似度'''
word1 = "郭靖"
word2 = "黄蓉"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")


word1 = "郭靖"
word2 = "杨康"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

word1 = "郭靖"
word2 = "小龙女"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

word1 = "郭靖"
word2 = "韦小宝"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")



