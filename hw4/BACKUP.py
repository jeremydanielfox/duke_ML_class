from cvxopt import solvers, matrix, spmatrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

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
    return (data,y)

# Training function for SVM that outputs the lambda vector and lambda_not scalar needed to predict for SVM
def train(x,y, kernel):
    # Use the cvxopt quadratic programming solver to get a solved version of the SVM quadratic program
    # First, arrange the variables into the form given on Piazza
    # Next, rearrange them to be in the form needed for the solver
    # Note, the solver takes in only floats, or it will break
    def solve_quadratic(X, Y, kernel):
        # Y = np.multiply(y,np.transpose(y))
        n = len(x)
        # X = np.zeros((n,n))
        # for i in range(n):
        #     for j in range(n):
        #         #X[i][j] = np.dot(np.transpose(x[i]),x[j])
        #         X[i][j] = kernel_function(x[i],x[j])
        # D = np.multiply(X,Y)
        I = np.identity(n)
        # Z = np.zeros((n,1))

        # # translation to solver format
        # y = map(lambda a: float(a),y)
        A = matrix.trans(matrix(y))

        G = matrix(-1 * I)
        h = matrix(np.zeros((n,1)))

        b = matrix(0.0)
        q = -1.0 * matrix(np.ones((n,1)))
        # P = matrix(D)
        # sol = solvers.qp(P,q,G,h,A,b)
        # return sol
        def makeD(X, Y, n, kernel):
            y = matrix(Y)*matrix.trans(matrix(Y))
            x = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    #x[i][j] = np.dot(X[i], X[j])
                    x[i][j] = kernel(X[i],X[j])
            x_matrix_cvx = matrix.trans(matrix(x))
            D = matrix(np.zeros((n,n)))
            for i in range(n):
                for j in range(n):
                    D[i, j] = y[i, j] * x_matrix_cvx[i, j]
            return D
        D = makeD(X,Y,n,kernel)
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
        print "all_alphas"
        print len(all_alphas)
        print all_alphas
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
        print "************LAMB*********"
        print "points"
        print points
        print "alphas"
        print alphas
        print (lamb,lamb_not)
        return (lamb,lamb_not)
    return calculate_lambdas(solve_quadratic(x,y,kernel),x,y)

def predict(datapoint, lamb, lambda_not):
    return lamb.dot(datapoint) + lambda_not

def true_positive_rate(prediction,actual):
    n = np.sum(actual)
    true_pos = 0.0
    for i in range(len(prediction)):
        if prediction[i] == 1 and actual[i] == 1:
            true_pos = true_pos +1
    return true_pos/n

def false_positive_rate(prediction,actual):
    n = 0.0
    false_pos = 0.0
    for i in range(len(actual)):
        if actual[i]==0:
            n += 1
        if actual[i]==0 and prediction[i] == 1:
            false_pos += 1
    return false_pos/n

def test_and_graph(data,labels,predict_function):

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
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        return roc_auc

    predictions = [predict_function(d) for d in data]
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
print "Jeremy"
# end purpose

lambdas = train(train_x,train_y, lambda x, z: np.dot(x,z))
lamb = lambdas[0]
lamb_not = lambdas[1]
for pair in test_data:
    point = pair[0]
    label = pair[1]
    print label == np.sign(predict(point,lamb,lamb_not))

# Use normal kernel on credit card data

data = read_and_process_credit('creditCard.csv',(0,1,2,3,4,5,6,7,8,9))
data = zip(data[0],data[1])
np.random.shuffle(data)
split = int(np.floor(len(data)/9))
train_data = data[0:split]
test_data = data[split+1:]
train_x, train_y = zip(*train_data)
test_x, test_y = zip(*test_data)
lambdas = train(train_x,train_y, lambda x, z: np.dot(x,z))
lamb = lambdas[0]
lamb_not = lambdas[1]

test_and_graph(test_x, test_y, lambda point: predict(point,lamb,lamb_not))

# Use gaussian kernel on credit card data





