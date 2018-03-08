from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv
import numpy as np
import os

model = Model('./model/994/model.npy')

trX, vaX, teX, trY, vaY, teY = sst_binary()


if not os.path.exists('features'):
    os.makedirs('features')

    trXt = model.transform(trX)
    vaXt = model.transform(vaX)
    teXt = model.transform(teX)

    np.save('features/trXt',trXt)
    np.save('features/vaXt',trXt)
    np.save('features/teXt',trXt)

else:
    trXt = np.load('features/trXt.npy')
    vaXt = np.load('features/vaXt.npy')
    teXt = np.load('features/teXt.npy')



full_rep_acc, c, nnotzero, coef, lg_model = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)

# visualize sentiment unit
sentiment_unit = trXt[:, 3984]
plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg')
plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos')
plt.legend()
plt.show()
