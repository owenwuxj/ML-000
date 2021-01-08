%%cython -a

import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray my_target_encoding():
    cdef cnp.ndarray[int] y = np.random.randint(2, size=(100, 1))
    cdef cnp.ndarray[int] x = np.random.randint(10, size=(100, 1))

    cdef cnp.ndarray[float] resultArray = np.zeros(x.length)
    cdef int i
    
    cdef valueDict = dict() # key:1~10->value:apperance w/ 1
    cdef countDict = dict() # key:1~10->value:apperance w/ 0 and 1   
    for i in x.length:
        y_value = x[i]  # 0 or 1
        x_value = y[i]  # 1 ~ 10
        if x[i] not in valueDict.keys():
            valueDict[x_value] = y_value
            countDict[x_value] = 1
        else: 
            valueDict[x_value] += y_value
            countDict[x_value] += 1
    
    for i in resultArray.size:
            resultArray[i] = (valueDict[x[i]] - y[i]) / (countDict[x[i]] - 1)
    
    return resultArray