# 相似性度量函数， 输入列向量, 归一化 0-1
from numpy import *
import numpy as np
from numpy import linalg as la

def getSigK(Sigma, k):
    '''
    输入：
        Sigma： 输入的奇异值向量
        k: 取前几个奇异值
    输出：(k,k)的矩阵
    '''
    eyeK = np.eye(k)
    return mat(eyeK * Sigma[:k])

def reBuild(U, Sigma, VT, k):
    '''
    使用前k个特征值重构数据
    '''
    Sigk = getSigK(Sigma, k)
    # 左行右列
    return mat(np.dot(np.dot(U[:,:k], Sigk), VT[: k,:]))

# 欧式距离计算相似度;    看成坐标系中两个点，来计算两点之间的距离
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))   # 再归一化

# 余弦相似度计算相似度;   看成坐标系中两个向量，来计算两向量之间的夹角
def cosSim(inA, inB):
    '''
    基于余弦相似性度量
    '''
    sim = float(inA.T* inB) / (la.norm(inA) * la.norm(inB))
    return 0.5 + 0.5 * sim  # 归一化

#    user对item的预测评分方法
#   SVD分解后的3个小矩阵、评分矩阵、相似度评价函数、学号、课程标号
def svdMethod(svdData, dataMat, simMeas, user, item):
    '''
    输入：
        见recommend函数
    输出：
        Score(double): user对item的评分
    算法流程：
        1. for item_other in allItem
        2. if haveBeenScore(item_other)
        3.    compute_Simliar_Score(item, item_other)
        4. return Score
    '''
    N = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, I_t = svdData

    # 按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值
    k = 0
    while sum(Sigma[:k]) < sum(Sigma) * 0.9:    # 奇异值平方和占比的阈值，一般取0.9。 (一般前10%甚至更少的奇异值的平方和 就占 全部奇异值的平方和的90%)
        k = k+ 1

    # 取前K个奇异值，输出：(k, k)的矩阵
    SigK = getSigK(Sigma, k)
    # 根据k的值将原始数据转换到k维空间(低维)
    itemFeature = dataMat.T * U[:,:k] * SigK.I  # itemFeature表示课程(item)在k维空间转换后的值

    for j in range(N):
        if dataMat[user,j] == 0 or j == item:   # 若对课程j的评分为0 或 j=item
            continue
        # 计算j和item的相似度sim
        sim = simMeas(itemFeature[item,:].T, itemFeature[j,:].T)

        ratSim = dataMat[user, j] * sim    # 用"item和j的相似度"乘以"用户对课程j的评分"，并求和

        simTotal += sim     # #对所有相似度求和
        ratSimTotal += ratSim

    if simTotal == 0:
        return 0
    return ratSimTotal / simTotal   # 得到对课程item的预测评分


def recommedCoursePerson(dataMat, user, N=7, simMeas=ecludSim, estMethod=svdMethod):
    """
    输入：
        dataMat(mat)(M,N): 评分矩阵.    == scoreMatrix
        use(int): 想推荐的用户id.
        N(int): 为用户推荐的未评分的课程个数
        simMeas(double): 两个特征向量的相似度评价函数
        estMethod(double)：推荐核心方法，计算user对item的预测评分方法
    输出：
        N * (item, 评分)： N个课程以及其的评分
    算法流程：
        1. 找到所有未评分的课程
        2. 若没有未评分课程，退出
        3. 遍历未评分课程
        4. 计算未评分课程与其他课程的相似性，得到一个预测打分
        5. 评分降序排列，取前N个输出
    """
    print(user)
    dataMat = mat(dataMat)

    unRatedItems = nonzero(dataMat[user,:].A == 0)[1]   # 建立一个用户未评分item的列表

    if len(unRatedItems) == 0:  # 如果都已经评过分，则退出
        return None

    U, Sigma, I_t = la.svd(dataMat) # SVD矩阵分解，得：左奇异矩阵U、sigma矩阵、右奇异矩阵I
    item_and_score = []
    for item in unRatedItems:   # 对于每个未评分的item，都计算其预测评分;  item是第几门课（课程的标号）
        # 计算未评分课程与其他课程的相似性，得到一个预测打分
        score = estMethod([U, Sigma, I_t], dataMat, simMeas, user, item)
        item_and_score.append((item, score))

    # 同上svdMethod（）
    k = 0
    while sum(Sigma[:k]) < sum(Sigma) * 0.9:
        k = k+ 1
    SigK = getSigK(Sigma, k)    # # 取前K个奇异值，输出：(k, k)的矩阵
    userFeature  = dataMat * I_t[:,:k] * SigK.I # 根据k的值将原始数据转换到k维空间(低维),userFeature表示课程(item)在k维空间转换后的值
    recomedUserVec = userFeature[user,:]

    user_and_score = []
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出【数据和数据下标】，一般用在 for 循环当中
    for idx, each in enumerate(userFeature):
        if user != idx:
            # cosSim()计算余弦相似度
            user_and_score.append((idx, cosSim(recomedUserVec.T, each.T)))

    # 按照item的得分进行从大到小排序
    recommedCourse = sorted(item_and_score, key=lambda k: k[1], reverse=True)[:min(N, len(item_and_score))]
    recommedPerson = sorted(user_and_score, key=lambda k: k[1], reverse=True)[:min(N, len(user_and_score))]

    print(recommedCourse)
    print(recommedPerson)

    return recommedCourse, recommedPerson


def toBarJson(data, dict2id):
    """

    :param data: [(0, 5.0), (1, 5.0), (2, 5.0)]
    :return::
    {
        "source": [
            [2.3, "计算机视觉"],
            [1.1, "自然语言处理"],
            [2.4, "高等数学"],
            [3.1, "线性代数"],
            [4.7, "计算机网络"],
            [5.1, "离散数学"]
        ]
     }
    """
    jsonData = {"source":[]}
    for each in data:
        unit = [each[1], dict2id[each[0]]]
        jsonData['source'].append(unit)
    return jsonData

def regularData(data, a, b):
    """
    功能，将列表的值归一化到[a,b]之间
    """
    dataNum = [i[0] for i in data['source']]
    Max, Min = max(dataNum), min(dataNum)
    k = (b-a)/(Max-Min)
    dataRg = [a+ k*(i-Min) for i in dataNum]
    for idx,each in enumerate(data['source']):
        each[0] = dataRg[idx]
    return data