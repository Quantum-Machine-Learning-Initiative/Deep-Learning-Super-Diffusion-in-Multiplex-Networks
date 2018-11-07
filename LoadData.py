import numpy as np
import h5py

# Dimension of the network (number of nodes in one layer, e.g. 50)
dimens = 50

PREFIX = "/full/path/to/.h5/folder/"

TRAINFILE="train.h5"
TESTFILE="test.h5"


# return the number of train instances avaible
def getMaxTrainSize():
    hf = h5py.File(PREFIX+TRAINFILE, 'r')
    ret = len(hf['train_p1'])
    hf.close()
    return ret

# return the number of test instances avaible
def getMaxTestSize():
    hf = h5py.File(PREFIX+TESTFILE, 'r')
    ret = len(hf['test_p1'])
    hf.close()
    return ret

# return the train examples with every component present as the i-th element of the relative array 
# input: size number of train instances to receive; if not given will retrieve all the instances present
# output:  p1, p2 connection probability in layer 1, 2
#          A1, A2 adjacency matrix of layer 1, 2
#          Y one-hot vector where 0 (no SD) is embed as [1, 0] and 1 (presence of SD) as [0, 1]
def getTrain(size=-1):

    # open the .h5 file 
    hf = h5py.File(PREFIX+TRAINFILE, 'r')

    # if no size is passed get all the elements
    if size<=0:
        size=len(hf['train_p1'])
    else:
    # if size is passed get only the elements requested
        real_size=len(hf['train_p1'])
        if real_size<size:
            print("The number of training examples requested (" + str(size) + ") is to high; avaible only " + str(real_size) + "!!!")
            size=real_size
            
    # retrive the instances
    p1 = np.array(hf['train_p1'][:size])
    p2 = np.array(hf['train_p2'][:size])
    A1 = np.array(hf['train_A1'][:size])
    A2 = np.array(hf['train_A2'][:size])
    Y_int =  np.array(hf['train_Y'][:size])
    Y =  np.zeros((size, 2))

    for i in range(size):
        Y[i] = np.array([1 if hf['train_Y'][i]==0 else 0, 1 if hf['train_Y'][i]==1 else 0])

    hf.close()
    return p1, p2, A1, A2, Y

# return the test examples with every component present as the i-th element of the relative array 
# input: size number of test instances to receive; if not given will retrieve all the instances present
# output:  p1, p2 connection probability in layer 1, 2
#          A1, A2 adjacency matrix of layer 1, 2
#          Y one-hot vector where 0 (no SD) is embed as [1, 0] and 1 (presence of SD) as [0, 1]
def getTest(size=-1):
    hf = h5py.File(PREFIX+TESTFILE, 'r')
    if size<=0:
        size=len(hf['test_p1'])
    else:
        real_size=len(hf['test_p1'])
        if real_size<size:
            print("The number of test examples requested (" + str(size) + ") is to high; avaible only " + str(real_size) + "!!!")
            size=real_size
    p1 = np.array(hf['test_p1'][:size])
    p2 = np.array(hf['test_p2'][:size])
    A1 = np.array(hf['test_A1'][:size])
    A2 = np.array(hf['test_A2'][:size])
    Y_int =  np.array(hf['test_Y'][:size])
    Y =  np.zeros((size, 2))

    for i in range(size):
        Y[i] = np.array([1 if hf['test_Y'][i]==0 else 0, 1 if hf['test_Y'][i]==1 else 0])

    hf.close()
    return p1, p2, A1, A2, Y
