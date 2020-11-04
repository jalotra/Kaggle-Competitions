
import random
import numpy as np
from pprint import pprint

def log_loss(y_pred, y_true):
    assert(len(y_pred) == len(y_true))
    loss = 0
    for (prediction, target) in zip(y_pred, y_true):
        loss += -1*(target*np.log10(prediction) + (1 - target)*np.log10(1 - prediction)) 
    
    return loss


if __name__ == "__main__":
    size = 10
    y_true = [random.randint(0, 1) for _ in range(size)]
    y_prediction = [random.randint(1,5) for _ in range(size)]
    total = sum(y_prediction)

    y_prediction = [x/np.sqrt(total) for x in y_prediction]
    loss = log_loss(y_prediction, y_true)
    pprint(f"The total Log loss for this combination is : {loss}") 