# Code by Jeremy Fox

from cvxopt import solvers, matrix, spmatrix
from sklearn import metrics, preprocessing, svm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hard solver

# Import data
def read_and_process(filename):
    # load in using fancy method. Note that first 50 are one class, next are another
    data = np.loadtxt(filename, delimiter=',', usecols=(0,1,2,3))
    # grab first hundred entries
    data = data[0:100]
    # construct labels
    y = np.array([1.0]*50 + [-1.0]*50)
    return (data,y)

def read_and_process_credit(filename, cols):
    data = np.loadtxt(filename, delimiter=',', usecols=cols)
    y = np.array([i[-1] for i in data])
    data = np.array(map(lambda a: np.delete(a,-1),data))
    # standardize
    data = preprocessing.scale(data)
    return (data,y)

# Training function for SVM that outputs the lambda vector and lambda_not scalar needed to predict for SVM
def train(x,y, kernel):
    # Use the cvxopt quadratic programming solver to get a solved version of the SVM quadratic program
    # First, arrange the variables into the form given on Piazza
    # Next, rearrange them to be in the form needed for the solver
    # Note, the solver takes in only floats, or it will break
    def solve_quadratic(X, Y, kernel):
        n = len(x)
        I = np.identity(n)
        A = matrix.trans(matrix(y))

        G = matrix(-1 * I)
        h = matrix(np.zeros((n,1)))

        b = matrix(0.0)
        q = -1.0 * matrix(np.ones((n,1)))

        yo = matrix(Y)*matrix.trans(matrix(Y))
        xo = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                xo[i][j] = kernel(X[i],X[j])
        x_mat = matrix.trans(matrix(xo))
        D = matrix(np.zeros((n,n)))
        for i in range(n):
            for j in range(n):
                D[i, j] = yo[i, j] * x_mat[i, j]

        sol = solvers.qp(D, q, G, h, A, b)
        return sol

    # takes in solution from quadratic solver
    # Returns a tuple of lambda and lambda_not
    def calculate_lambdas(sol,x,y):
        d = len(x[0])
        lamb = np.zeros((1,d))
        lamb_not = 0
        points = []
        all_alphas = sol['x']
        sorted(all_alphas)
        alphas=[]
        threshold = .0001
        for i, alpha in enumerate(all_alphas):
            if alpha > threshold:
                alphas+=[alpha]
                points+=[x[i]]
                lamb += alpha * y[i] * x[i]
        for i, alpha in enumerate(all_alphas):
            if alpha > threshold and y[i] == 1:
                lamb_not = 1 - np.dot(lamb,x[i])
                break

        return (lamb,lamb_not)
    return calculate_lambdas(solve_quadratic(x,y,kernel),x,y)

def predict(datapoint, lamb, lambda_not):
    return lamb.dot(datapoint) + lambda_not

def test_and_graph(data,labels,predict_function,title):

    def roc(prediction,actual):
        fpr, tpr, other = metrics.roc_curve(actual, prediction)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        return roc_auc

    def accuracy(predictions, actual):
        n = float(len(actual))
        actual = [1.0 if i==1.0 else -1.0 for i in actual]
        predictions = np.sign(predictions)
        cnt = 0.0
        for idx, val in enumerate(actual):
            if val == predictions[idx]:
                cnt+=1
        return cnt/n


    labels = list(labels)
    predictions = [predict_function(d).tolist() for d in data]
    predictions = [item for sublist in predictions for item in sublist]

    auc = roc(predictions,labels)
    return auc


# Use normal kernel on iris data
data = read_and_process('iris.data.txt')
XX = data[0]
YY = data[1]
data = zip(data[0],data[1])
np.random.shuffle(data)
train_data = data[0:90]
test_data = data[91:]
test_x, test_y = zip(*test_data)
train_x, train_y = zip(*train_data)

# for purpose of printing out lambdas
lambdas = train(XX,YY, lambda x, z: np.dot(np.transpose(x),z))
# end purpose

lambdas = train(train_x,train_y, lambda x, z: np.dot(x,z))
lamb = lambdas[0]
lamb_not = lambdas[1]

# Use normal kernel on credit card data

data = read_and_process_credit('creditCard.csv',(0,1,2,3,4,5,6,7,8,9))
data = zip(data[0],data[1])
np.random.shuffle(data)
split = int(np.floor(len(data)/9))
train_data = data[0:split]
test_data = data[split+1:]
train_x, train_y = zip(*train_data)
test_x, test_y = zip(*test_data)


linear_clf = svm.SVC(kernel='linear')
linear_clf.fit(train_x,train_y)
test_and_graph(test_x, test_y, linear_clf.decision_function,"Linear Kernel SVM")

# Use radial base kernel on credit card data

data = read_and_process_credit('creditCard.csv',(0,1,2,3,4,5,6,7,8,9))
data = zip(data[0],data[1])
np.random.shuffle(data)
split = int(np.floor(len(data)/9))
train_data = data[0:split]
test_data = data[split+1:]
train_x, train_y = zip(*train_data)
test_x, test_y = zip(*test_data)

# sigma^2 = 2
gamma_val = .5
radial_clf = svm.SVC(kernel='rbf',gamma=gamma_val)
radial_clf.fit(train_x,train_y)
test_and_graph(test_x, test_y, radial_clf.decision_function,"Radial Base Kernel, $\\sigma^2=2$")

# sigma^2 = 20

gamma_val=.05
radial_clf = svm.SVC(kernel='rbf',gamma=gamma_val)
radial_clf.fit(train_x,train_y)
test_and_graph(test_x, test_y, radial_clf.decision_function, "Radial Base Kernel, $\\sigma^2=20$")


