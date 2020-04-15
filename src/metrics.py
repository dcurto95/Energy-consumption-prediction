from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import preprocessing


def compute_print_scores(prediction, test_y):
    real_mse = mean_squared_error(test_y, prediction)
    mape = np.mean(np.abs((test_y - prediction) / test_y)) * 100
    r2test = r2_score(test_y, prediction)
    print('Real MSE =', real_mse)
    print('MAPE(%)', mape)
    print('R2 test= ', r2test)

    return real_mse, mape, r2test
