# -*- coding: utf-8 -*-

import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence, w2v_model):
    word_vectors = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
    # 取平均值作为段落向量
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

#预处理函数
def pretreatment_para(paragraph):
    # 预处理
    paragraph = paragraph.replace('\n', '')  # 去除换行符
    paragraph = paragraph.replace(' ', '')  # 去除空格
    paragraph = paragraph.replace('\u3000', '')  # 去除全角空白符
    paragraph = paragraph.replace('，', '')
    # 分词
    words = jieba.cut(paragraph)
    return words
    # 计算词向量
model = Word2Vec.load(r'D:\word2vec.model')  # 加载模型，由generate_model.py生成
paragraph1 = "宗维侠强道：“七伤拳是我崆峒绝技，怎能说有害无益？当年我掌门师祖木灵子以七伤拳威震天下，名扬四海，寿至九十一岁，怎么说会损害自身？你这不是胡说八道麽？”张无忌道：“木灵子前辈想必内功深湛，自然能练，不但无害，反而强壮肝腑。依晚辈之见，宗前辈的内功如不到那个境界，若要强练，只怕终归无用。”"
#paragraph2 = "宗维侠是崆峒名宿，虽知他所说的不无有理，但在各派高手之前，被这少年指摘本派的镇山绝技无用，如何不恼？大声喝道：“凭你也配说我崆峒绝技有用无用。既说无用，那就来试试。”张无忌淡淡一笑，说道：七伤拳自是神妙精奥的绝技，拳力刚中有柔，柔中有刚，七般拳劲各不相同，吞吐闪烁，变幻百端，敌手委实难防难挡……”宗维侠听他赞誉七伤拳的神妙，说来语语中肯，不禁脸露微笑，不住点头，却听他继续说道：“……晚辈只是说内功修为倘若不到，那便练之有害无益。”"
#paragraph2 = "原来黄药师对妻子情深意重，兼之爱妻为他而死，当时一意便要以死相殉。他自知武功深湛，上吊服毒，一时都不得便死，死了之后，尸身又不免受岛上哑仆糟蹋，于是去大陆捕拿造船巧匠，打造了这艘花船。这船的龙骨和寻常船只无异，但船底木材却并非用铁钉钉结，而是以生胶绳索胶缠在一起，泊在港中之时固是一艘极为华丽的花船，但如驶入大海，给浪涛一打，必致沉没。他本拟将妻子遗体放入船中，驾船出海，当波涌舟碎之际，按玉箫吹起《碧海潮生曲》，与妻子一齐葬身万丈洪涛之中，如此潇洒倜傥以终此一生，方不辱没了当世武学大宗匠的身分，但每次临到出海，总是既不忍携女同行，又不忍将她抛下不顾，终于造了墓室，先将妻子的棺木厝下。这艘船却是每年油漆，历时常新。要待女儿长大，有了妥善归宿，再行此事。"
paragraph2 = "这时，李达康的专车也驶临了收费站。侯亮平下车，当道一站，举起手掌做出停车的手势。李达康的轿车缓缓停下。后面跟踪而来的张华华的警车也在李达康专车的左侧停了下来。李达康的司机下了车，走到侯亮平和陆亦可面前：你们想干啥？知道这是谁的车吗？侯亮平说：我只知道被传唤人欧阳菁在这台车上。司机满脸的不屑：我说同志，欧阳菁是谁的夫人，你不会不知道吧？侯亮平说：欧阳菁是谁的夫人与我们检察院办案没关系！陆亦可解释道：有人举报了欧阳菁副行长，我们要请她去谈一谈！司机不无傲慢地说：知道吗？欧阳副行长是市委李达康书记的夫人，这台车也是李达康书记的专车！"
paragraph1 = pretreatment_para(paragraph1)
paragraph2 = pretreatment_para(paragraph2)

vector1 = build_sentence_vector(paragraph1, model)
vector2 = build_sentence_vector(paragraph2, model)
# 计算段落相似度
if np.all(vector1 == 0) or np.all(vector2 == 0):
    paragraph_similarity_score = 0  # 若有一个段落无有效向量，则相似度为0
else:
    paragraph_similarity_score = cosine_similarity([vector1], [vector2])[0][0]
print(f"段落间的语义相似度：{paragraph_similarity_score}")