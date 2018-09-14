#coding:utf-8
#_author_='PYJ'

#coding:utf-8
#_author_='PYJ'

import numpy as np

def readFile(path):
    f = open(path,'rb')
    answerList=[];i=0
    marklist=[]
    mark=[]
    dataset=[]
    for line in f.readlines():  # 逐行进行处理
        data = line.decode().strip('\r\n')
        nums = data.split("\t")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        dataset.append([nums[0],nums[1]])
        mark.append(nums[2])
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
    return dataset,mark

#test
if __name__ == '__main__':
    readFile("E:\毕业设计\毕业设计\DataSet\Aggregation.txt")