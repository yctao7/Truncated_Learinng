"""
Truncated logistic(probit) regression
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

list01 = [-2, -1, -0.5, -0.3,-0.2 ,-0.1] # Some values of predetermined C
liststd = []
listtrunc = []
k = 1000 # loop count

for c in list01:
    stdsim = []
    stdmean = []
    truncsim = []
    truncmean = []
    for i in range(k):

        # Data generation process

        m = 10000 # number of observations
        d = 2 # dimension
        X = np.random.uniform(0, 100, size=m)
        X = X.reshape(-1, 1)
        for i in range(d-1):
            x = np.random.uniform(0, 100, size=m)
            x = x.reshape(-1, 1)
            X = np.hstack([X, x])


        # True parameter
        theta1 = np.random.uniform(-1, 1, size=2)
        b = 0.0 # Intercept
        z = X.dot(theta1) + b + np.random.logistic(0, 1, size=m)
        # z = X.dot(theta1) + b + np.random.normal(0, 1, size=m) # probit


        # Truncation process
        ztrunc = []
        for i in list(np.arange(X.shape[0])):
            if z[i] > c:
                ztrunc.append(z[i])

        xtrunc = []
        for i in list(np.arange(X.shape[0])):
            if z[i] > c:
                xtrunc.append(X[i])

        # Guarantee the existence of training sample
        if len(xtrunc) <= 1:
            continue

        # Label
        y = []
        for i in ztrunc:
            if i > 0:
                y.append(1)
            else:
                y.append(0)

        ztrunc = np.array(ztrunc)
        xtrunc = np.array(xtrunc)
        y = np.array(y)

        # Sigmoid function
        def sig(x):
            return 1 / (1 + np.exp(-x))


        # Gradient calculation
        def grad(theta, X_b_i, y_i):
            x = X_b_i.dot(theta)
            return X_b_i.T.dot(sig(x) - y_i)
            # return  X_b_i.T.dot(norm.cdf(x,0,1) - y_i) # probit
            
        # Standard SGD
        def sgd(X, y, initial_theta, iters):

            t0, t1 = 5, 5000
            def learning_rate(t): # learning rate
                return t0 / (t + t1)

            rand = []

            theta = initial_theta
            for cur_iter in range(iters):
                rand_i = np.random.randint(len(X)) # random choice of a single observation

                if rand_i in rand:
                    continue
                else:
                    rand.append(rand_i)

                gradient = grad(theta, X[rand_i], y[rand_i])
                theta = theta - learning_rate(cur_iter) * gradient # parameter updated

            return theta

        X_b = np.hstack([np.ones((len(xtrunc), 1)), xtrunc])
        initial_theta = np.zeros(X_b.shape[1])
        theta = sgd(X_b, y, initial_theta, iters=1000)


        # Cosine similarity
        def cosine_sim(theta1, theta2):
            sim = theta1.dot(theta2)/(np.linalg.norm(theta1)*np.linalg.norm(theta2))
            return sim

        stdsim.append(cosine_sim(theta[1:3], theta1))
        stdmean.append(np.mean(stdsim))


        def truncgrad(theta, X_b_i, y_i):
            x = X_b_i.T.dot(theta)
            return X_b_i.T.dot(sig(x) + (sig(c-x) - 1) * y_i)
            # return  X_b_i.T.dot(norm.cdf(x, 0, 1) + (norm.cdf(c-x, 0, 1) - 1) * y_i) # probit
            

        # Truncated SGD
        def truncsgd(X_b, y, initial_theta, n_iters):

            t0, t1 = 5, 5000

            def learning_rate(t): 
                return t0 / (t + t1)

            rand = []

            theta = initial_theta
            for cur_iter in range(n_iters):
                rand_i = np.random.randint(len(X_b))

                if rand_i in rand:
                    continue
                else:
                    rand.append(rand_i)

                gradient = truncgrad(theta, X_b[rand_i], y[rand_i])
                theta = theta - learning_rate(cur_iter) * gradient

            return theta

        X_b = np.hstack([np.ones((len(xtrunc), 1)), xtrunc])
        initial_theta = np.zeros(X_b.shape[1])
        theta = truncsgd(X_b, y, initial_theta, n_iters=1000)
        

        truncsim.append(cosine_sim(theta[1:3], theta1))
        truncmean.append(np.mean(truncsim))

    liststd.append(stdmean[-1])
    listtrunc.append(truncmean[-1])

# Plot
fig, ax = plt.subplots()
ax.plot(list01, liststd, label='Standard SGD', color='b', marker='s')
ax.plot(list01, listtrunc, label='Truncated SGD', color='r', marker='s')

plt.legend()
plt.grid()
plt.show()



