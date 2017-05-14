# Author: Mathieu Ré

import random
import numpy as np

def bootstrap (test, label, score, predict, p=100):
    ''' Gives a score according to the Bootstrap method
    @var test : Testing set as a list or array
    @var label : Labels on the testing set as a list or array
    @var score : score metric : (y_true, y_pred) -> score
    @var predict : desicion function to evaluate
    @returns : bootstrap score
    '''
    scores = []
    N_test = len(test)
    for i in range(p):
        test_sub = []
        index_sub = []
        label_sub = []
        for k in range(N_test):
            ind = random.randint(0, N_test-1)
            index_sub.append(ind)
            test_sub.append(test[ind])
            try :
                label_sub.append(label[ind])
            except IndexError:
                print("Lol : %d"%ind)
                break
        scores.append(score(label_sub,predict(test_sub)))
    return np.std(scores)
