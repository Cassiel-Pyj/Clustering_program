import numpy as np

def readFile(path):
    f = open(path,'rb')
    answerList=[];i=0
    marklist=[]
    dataset=[]
    for line in f.readlines():  # 逐行进行处理
        data = line.decode().strip('\r\n')
        nums = data.split("  ")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        dataset.append([nums[0],nums[1]])
    dataset=np.mat(dataset)
    f.close()
    return dataset

if __name__ == '__main__':
    dataset=readFile("E:\毕业设计\毕业设计\DataSet\GuassMixpp.txt")
    print(dataset)