import codecs

from numpy import *
import numpy as np
import operator
# listdir可以列出给定目录的文件名
from os import listdir
from collections import Counter


def createDataSet():
    # 创建数据集和标签
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量
    :param k:选择最近邻居的数目
    注：labels元素数目和dataSet行数相同；程序使用欧式距离公式
    """
    # -----------实现 classify0() 方法的第一种方式--------------#
    # 1. 距离计算
    # 输出训练集样本矩阵的行数
    dataSetSize = dataSet.shape[0]

    # tile生成和训练样本对应的矩阵，并与训练样本求差
    # tile(向量，（x,y))：将向量在行方向上重复x次，在列方向上重复y次
    # 下方代码则是将向量在行方向上重复训练集样本矩阵的行数，重复一列；然后减去训练样本集
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 这个乘方是数组里边的各个数分别乘方
    sqDiffMat = diffMat ** 2
    # 将矩阵每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 这个distances好像是数组
    distances = sqDistances ** 0.5
    # 根据距离按照从小到大排序，然后返回对应的索引位置
    # argsort()是将x中的元素按照从小到大排序并且返回对应的索引值位置
    sortedDistIndicies = distances.argsort()

    # 2. 选择距离最小的k个点
    # 字典
    classCount = {}
    # rang(k):数在0到k-1之间
    for i in range(k):
        # 找到样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]

        # 若该类型存在，则返回该类型(key)所对应的值(value)并在字典classCount将该类型对应的值+1；若不存在则返回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 3. 排序并返回出现最多的那个类型-利用max函数直接返回字典中value最大对应的key值
    maxclassCount = max(classCount, key=classCount.get)
    return maxclassCount


def test1():
    group, labels = createDataSet()
    print('str(group)=', str(group))
    print('str(labels=', str(labels))
    # 输出出现次数最多的那个类型
    print('max=', classify0([0.1, 0.1], group, labels, 3))

# ------------------------------上边为简单的方法--------------------------------------------#


def file2matrix(filename):
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应空矩阵
    # 例如：zeros((2，3))就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = codecs.open(filename, 'r', 'utf-8')
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # '\t':代表空四个字符
        # 修改了此处，文本格式不匹配'\t'，只能用' '
        listFromLine = line.split(' ')
        # 将读取到的字符串数组listFromLine中的字符串从0到2赋值给returnMat中的每一行
        returnMat[index, :] = listFromLine[0:3]
        # 数据中每一行的最后一个是label标签数据，即每一行所对应的类别，将该类别存储到classLabelVector中
        # 因为读取到的数据是字符串，所以要加int
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        # returnMat为把文件中的数据划分好的矩阵，classLabelVector为矩阵每一行所代表的数据类别
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
       归一化特征值，消除属性之间量级不同导致的影响
       :param dataSet: 数据集
       :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
       归一化公式：
           Y = (X-Xmin)/(Xmax-Xmin)
           其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # min(0):每一列的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # m是dataSet矩阵的行数
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # 用归一化后的矩阵来作为用于分类的inX
    return normDataSet, ranges, minVals


def datingClassTest():
    """
       对约会网站的测试方法
       :return: 错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    # 测试范围,一部分测试一部分作为样本
    hoRatio = 0.1
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('E:\Python\AiLearning\datingTestSet2.txt')
    # 归一化数据:normMat是归一化后的标准数组，ranges是极差
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 计算测试样本的数量
    numTestVecs = int(m * hoRatio)
    print('测试样本的数量=', numTestVecs)
    errorCount = 0.0
    # 让测试数据集每一次都要与训练数据进行比较：i是测试数据集，numTestVecs:m是训练数据集
    for i in range(numTestVecs):
        # 因为numTestVecs之前的都是测试样本的数量，numTestVecs:m是训练样本的数量
        #datingLabels是数组而不是矩阵
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("错误个数是：", errorCount)


if __name__ == '__main__':
    datingClassTest()
