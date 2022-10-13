from numpy import *
import operator

from sklearn import datasets

def createDataSet():
    """创建数据集和标签

    Returns:
        group : 日志文件中不同的测量点或者入口
        labels : 每个数据点的标签信息
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """使用k-近邻算法将每组数据划分到某个类中

    Args:
        inX : 用于分类的输入向量
        dataSet : 输入的训练样本集
        labels : 标签向量
        k : 选择最近邻居的数目
    
    Returns:
        sortedClassCount[0][0] : 
    """
    # 距离计算
    dataSetSize = dataSet.shape[0]        # 训练样本集行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)  
    # array.sum(axis = 0),对array的每一列进行相加  axis =1,对array的每一行进行相加
    # array.sum(),对array的全部元素进行相加
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()
    # argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
    # 排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # python2 iteritems() python3 items()
    
    return sortedClassCount[0][0]
        