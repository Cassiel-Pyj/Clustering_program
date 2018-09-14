#coding:utf-8
#_author_='PYJ'

import numpy as np

def readFile(path):
    f = open(path)
    answerList=[]
    marklist=[]
    dataset=[]
    for line in f.readlines():  # 逐行进行处理
        data = line.strip('\n')
        nums = data.split(",")
        nums = [float(x) for x in nums]
        dataset.append([nums[0],nums[1]])
        if nums[2] not in marklist:
            marklist.append(nums[2])
            answerList.append([[nums[0],nums[1]]])
        else:
            index=np.inf
            for i in range(len(marklist)):
                if nums[2]==marklist[i]:
                    index=i
                    break
            answerList[index].append([nums[0],nums[1]])
    dataset=np.mat(dataset)
    # print("dataset",dataset)
    # print("answerList",answerList)
    # print("答案列表长度",len(answerList))
    f.close()
    return dataset,answerList

def readUCI(path):
    f=open(path,'rb')
    lines=f.readlines()
    dataset=[]
    marklist=[]
    answerList=[]
    for line in lines:
        data = line.decode().strip('\r\n')
        nums = data.split(",")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        # print(nums)
        dataset.append([float(x) for x in nums[1:len(nums)]])
        # print(dataset)
        if nums[0] not in marklist:
            marklist.append(nums[0])
            answerList.append([[float(x) for x in nums[1:len(nums)]]])
        else:
            index=np.inf
            for i in range(len(marklist)):
                if nums[0]==marklist[i]:
                    index=i
                    break
            answerList[index].append([float(x) for x in nums[1:len(nums)]])
    dataset=np.array(dataset)
    # print(dataset)
    # print(answerMark)
    return dataset,answerList
    f.close()

#test
if __name__ == '__main__':
    dataset,answerClusterMark=readUCI("E:\毕业设计\毕业设计\DataSet\Data_Iris.csv")
    print(dataset,answerClusterMark)
