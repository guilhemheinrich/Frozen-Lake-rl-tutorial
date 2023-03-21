from nptyping import Float, NDArray, Shape
import numpy as np

def reshape(Q_sa: NDArray[Shape['StateDimension, ActionDimension'], Float]):
    reshaped_Q_sa = np.zeros(Q_sa.shape)
    num_rows, num_cols = Q_sa.shape
    for row_index in range(num_rows):   
        print(int(np.argmax(Q_sa[row_index,])))   
        reshaped_Q_sa[row_index, int(np.argmax(Q_sa[row_index,]))] = 1
    return reshaped_Q_sa


q = np.random.random((10, 3))
q_shape = reshape(q)
print(q)
print(q_shape)