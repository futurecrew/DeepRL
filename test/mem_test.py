import numpy as np

def test1():
    big_list = []
    list = []
    
    for i in range(10):
        a = np.zeros((3, 3), dtype=np.int)    
        a.fill(i)
        
        if len(list) == 4:
            del list[0]
        list.append(a)
        
        if len(list) == 4:
            dataList = []
            for j in range(4):
                dataList.append(list[j])
            big_list.append(dataList)


test1()