"""
    Task from https://bayesgroup.github.io/deepbayes-school/2017/task/description/
"""

# -*- coding: utf-8 -*-

import numpy as np
from scipy import special
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import f1_score

# Используйте scipy.special для вычисления численно неустойчивых функций
# https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special

def compute_gradient(F, x):
    """
        Computing gradient F in x using approximate formula.

        Parameters
        -------
            F : functuion from Rn to R
            x : point if Rn

        Returns
            Gradient vector
    """
    grad = np.zeros(x.shape)
    h = 0.00001

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]

        x[ix] = old_value - h
        fxh_m = F(x)

        x[ix] = old_value + h
        fxh_p = F(x)

        x[ix] = old_value
        grad[ix] = (fxh_p - fxh_m) / (2 * h)
        it.iternext()

    return grad

def lossf(w, X, y, l1, l2):
    """
    Вычисление функции потерь.

    :param w: numpy.array размера  (M,) dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: float, значение функции потерь
    """
    powers = -1 * y * np.dot(X, w)

    # array of log(e^0 + e^powers[i])
    losses = np.logaddexp(0, powers)

    r1 = l1 * np.sum(np.abs(w))
    r2 = l2 * np.sum(w ** 2)
    lossf = np.sum(losses) + r1 + r2
    return lossf

def gradf(w, X, y, l1, l2):
    """
    Вычисление градиента функции потерь.

    :param w: numpy.array размера  (M,), dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: numpy.array размера  (M,), dtype = np.float, градиент функции потерь d lossf / dw
    """
    log_loss = np.dot(-y * special.expit(-y * X.dot(w)), X)
    gradw = log_loss + l1 * np.sign(w) + 2 * l2 * w
    return gradw

def grads_error(num_grad, analitic_grad):
    """
        Calculates error for two gradients.

        :param num_grad: numpy.array, dtype=np.float,
                         numerical calculated gradient
        :param analitic_grad: numpy.array, dtype=np.float,
                              analitically calculated gradient
    """
    diff_norm = np.linalg.norm((num_grad - analitic_grad), ord=1)
    num_norm = np.linalg.norm(num_grad, ord=1)
    analitic_norm = np.linalg.norm(analitic_grad, ord=1)

    return diff_norm / max(num_norm, analitic_norm)

class LR(ClassifierMixin, BaseEstimator):
    def __init__(self, lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=0):
        """
        Создание класса для лог регрессии
        
        :param lr: float, длина шага для оптимизатора
        :param l1: float, l1 коэффициент регуляризатора 
        :param l2: float, l2 коэффициент регуляризатора
        :param num_iter: int, число итераций оптимизатора
        :param verbose: bool, ключик для вывода
        """
        self.l1 = l1
        self.l2 = l2
        self.lr = lr
        self.verbose = verbose
        self.num_iter = num_iter
        self.mean = 0

    def preprocess(self, X):
        """
            Preprocess data.

            :param X: numpy.array (N, M), dtype = np.float
            :return: X
        """
        X -= self.mean

        # X /= np.std(X, axis = 0)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X = min_max_scaler.fit_transform(X)

        return X

    def fit(self, X, y):
        """
        Обучение логистической регрессии.
        Настраивает self.w коэффициенты модели.

        Если self.verbose == True, то выводите значение 
        функции потерь на итерациях метода оптимизации. 

        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :return: self
        """
        n, d = X.shape
        self.mean = np.mean(X, axis=0)
        self.w = 0.01 * np.random.randn(d)
        X = self.preprocess(X)

        for i in xrange(self.num_iter):
            loss = lossf(self.w, X, y, self.l1, self.l2)

            gradient = gradf(self.w, X, y, self.l1, self.l2)

            if self.verbose and i % 100 == 0:
                print 'Loss = %f' % loss
                gradient_n = compute_gradient(lambda weights: lossf(weights, X, y, self.l1, self.l2),
                                              self.w)
                err = grads_error(gradient_n, gradient)
                print 'Error = %f' % err

            self.w += -self.lr * gradient

        return self

    def predict_proba(self, X):
        """
        Предсказание вероятности принадлежности объекта к классу 1.
        Возвращает np.array размера (N,) чисел в отрезке от 0 до 1.

        :param X: numpy.array размера  (N, M), dtype = np.float
        :return: numpy.array размера  (N,), dtype = np.float
        """
        X = self.preprocess(X)
        responses = np.dot(X, self.w)
        predicts = special.expit(responses)

        return predicts

    def predict(self, X):
        """
        Предсказание класса для объекта.
        Возвращает np.array размера (N,) элементов 1 или -1.

        :param X: numpy.array размера  (N, M), dtype = np.float
        :return:  numpy.array размера  (N,), dtype = np.int
        """
        # Вычислите предсказания для каждого объекта из X
        X = self.preprocess(X)
        predicts = self.predict_proba(X)

        mask_1 = predicts > 0.5
        predicts[:] = -1
        predicts[mask_1] = 1

        return predicts

def test_work():
    print "Start test"
    X, y = make_classification(n_features=100, n_samples=1000)
    y = 2 * (y - 0.5)

    X_test, y_test = make_classification(n_features=100, n_samples=1000)
    y_test = 2 * (y - 0.5)

    try:
        clf = LR(lr=1e-3, l1=1e-4, l2=1e-4, num_iter=1000, verbose=1)
    except Exception:
        assert False, "Создание модели завершается с ошибкой"
        return

    try:
        clf = clf.fit(X, y)
    except Exception:
        assert False, "Обучение модели завершается с ошибкой"
        return

    predicts = clf.predict(X_test)
    accuracy = np.mean(y_test == predicts)
    f1 = f1_score(y_test, predicts)
    print 'Accuracy after training = %f' % accuracy
    print 'F1-Score = %f' % f1

    logreg = linear_model.LogisticRegression(C=1e-4, max_iter=1000)
    logreg.fit(X, y)
    sklrn_prd = logreg.predict(X_test)
    sklrn_acc = np.mean(y_test == sklrn_prd)
    sklrn_f1_score = f1_score(y_test, sklrn_prd)
    print 'SkLearn logistic regressuion accuracy = %f' % sklrn_acc
    print 'SkLearn F1-Score = %f' % sklrn_f1_score

    assert isinstance(lossf(clf.w, X, y, 1e-3, 1e-3), float), "Функция потерь должна быть скалярной и иметь тип np.float"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).shape == (100,), "Размерность градиента должна совпадать с числом параметров"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).dtype == np.float, "Вектор градиента, должен состоять из элементов типа np.float"
    assert clf.predict(X).shape == (1000,), "Размер вектора предсказаний, должен совпадать с количеством объектов"
    assert np.min(clf.predict_proba(X)) >= 0, "Вероятности должны быть не меньше, чем 0"
    assert np.max(clf.predict_proba(X)) <= 1, "Вероятности должны быть не больше, чем 1"
    assert len(set(clf.predict(X))) == 2, "Метод предсказывает больше чем 2 класса на двух классовой задаче"
    print "End tests"

test_work()
