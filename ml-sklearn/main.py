#%%

from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1)
mnist.keys()

#%%

x,y=mnist['data'],mnist['target']
x.shape
y.shape

#%% muestra los digitos

import matplotlib as mpl
from matplotlib import pyplot as plt
some_digits=x[1]
some_digit_image=some_digits.reshape(28,28)
plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()
y[1]




#%% Division de Set de test y set de entrenamiento

x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]

#%% clasificador Binario , Tratar de determinr el 5 con Stocastic gradient Descent

y_train_five=(y_train==5)
y_test_five=(y_test==5)



#%%

from sklearn.linear_model import SGDClassifier
sgd_clf =SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
# sgd_clf.fit(x_train,y_train_five)
# sgd_clf.predict([some_digits])



