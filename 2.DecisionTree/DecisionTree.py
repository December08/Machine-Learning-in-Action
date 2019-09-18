import copy
import operator
from math import log
import decisionTreePlot as dtPlot



def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    """
    # 求list的长度，表示参与训练的数据量
    numEntries = len(dataSet)
    # 计算分类标签label出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 对于Label标签的占比，求出Label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵，以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, index, value):
    """
        splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
            就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
        Args:
            dataSet 数据集                 待划分的数据集
            index 表示每一行的index列        划分数据集的特征
            value 表示index列对应的value值   需要返回的特征的值。
        Returns:
            index列为value的数据集【该数据集需要排除index列】
    """
    # 切分数据集
    # retDataSet中只装着featVec[index] == value的数据
    retDataSet = []
    for featVec in dataSet:
        # index列为value的数据集【该数据集需要排除index列】
        # 判断index列的值是否为value
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            """
                        请百度查询一下： extend和append的区别
                        extend:延长  append:在后边添加
                        list.append(object) 向列表中添加一个对象object
                        list.extend(sequence) 把一个序列seq的内容添加到列表中
                        1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
                        2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
                        result = []
                        result.extend([1,2,3])
                        print result
                        result.append([4,5,6])
                        print result
                        result.extend([7,8,9])
                        print result
                        结果：
                        [1, 2, 3]
                        [1, 2, 3, [4, 5, 6]]
                        [1, 2, 3, [4, 5, 6], 7, 8, 9]
            """
            reducedFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reducedFeatVec)
    # 返回index列为value的数据集【该数据集需要排除index列】
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征去切割数据)
       Args:
           dataSet 数据集
       Returns:
           bestFeature 最优的特征列
    """
    # 求第一行有多少列的 Feature, 最后一列是label列因此不计入Feature的数量中
    numFeatures = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Feature编号
    bestInfoGain, bestFeature = 0.0, -1
    # 对所有的特征值进行迭代
    for i in range(numFeatures):
        # 获取每一个实例的每一个feature，组成list集合
        featList = [example[i] for example in dataSet]
        # 获取剔重后的集合————即使用set对list数据进行去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            # 数据集每一行若第i列的值等于value,则将数据集划分为[0,i-1],i,[i+1,n]
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
       majorityCnt(选择出现次数最多的一个结果)
       Args:
           classList label列的集合
       Returns:
           bestFeature 最优的特征列
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    # key=operator.itemgetter(1)即通过classCount的第二个域——即label出现次数进行排序
    """
        输入：
        students = [('john', 'A', 8), ('jane', 'B', 12), ('dave', 'B', 10)]
        b =sorted(students, key=operator.itemgetter(2))
        print(b)
        输出：
        [('john', 'A', 8), ('dave', 'B', 10), ('jane', 'B', 12)]
    """
    # items()方法是将字典中的每个项分别做为元组，添加到一个列表中，形成了一个新的列表容器
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label含义——获取label的位置
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # print('bestFeatLabel:', bestFeatLabel)
    # 初始化myTree
    # 嵌套字典，bestFeatLabel: {}中包含的关键字就是bestFeatLabel的两个子节点
    myTree = {bestFeatLabel: {}}
    # print('初始化处', myTree)
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch【即最优列】做分类
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉featValues中重复的特征对应的特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性，在每个数据集划分上递归调用createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #                                                                                                           print('value=', value, myTree[bestFeatLabel][value])
    # print('结束循环后：', myTree)
    '''
        变量myTree包含了很多代表树结构信息的嵌套字典，从左边开始，第一个关键字no surfacing是第一个划分数据集的特征名称，该关键字的值
        也是另一个数据字典。第二个关键字是no surfacing特征划分的数据集，这些关键字的值是no surfacing节点的子节点。这些值可能是类标签，
        也可能是另一个数据字典。如果值是类标签，则该子节点是叶子结点，如果值是另一个数据字典，则子节点是一个判断节点，这种格式不断重复
        就构成了整棵树。
    '''
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    classify(给输入的节点，进行分类)
       Args:
           inputTree  决策树模型
           featLabels Feature标签对应的名称
           testVec    测试输入的数据
       Returns:
           classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree对应的所有Key
    firstKey = list(inputTree.keys())
    # 获取tree的根节点对应的key值
    firstStr = firstKey[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型【字典类型】
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 0]))
    dtPlot.createPlot(myTree)

if __name__ == "__main__":
    fishTest()
