#EECS545 Team project truncated linear regression
#formal for syn data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


C=0.5# Truncation threshold C
for C in [0.5]:
    N=80# number of observed data
    d=1#dimension of data
    theta_star=np.array([1/np.sqrt(d)]*d)
    np.random.seed(1)
    print("theta_star=",theta_star)
    #truncation function,when x<C,x will be truncated
    def phi(x):
        if x >= C:
            return 1
        else:
            return 0

    #generate truncated data
    def tru_data_generate(N):
        num = 0
        x_dataset = []
        y_dataset = []
        x_dataset_tru = []
        y_dataset_tru = []
        while ( num < N ):
            #new datapoint
            x = np.random.randn(d)
            e = np.random.randn()
            y = np.dot(theta_star,x) + e 
            
            #truncate
            if phi(y):
                x_dataset.append(x)
                y_dataset.append(y)
                #print("x=",x)
                #print("e=",e)
                #print("y=",y)
                num += 1
            else:
                x_dataset_tru.append(x)
                y_dataset_tru.append(y)
                
        return x_dataset,y_dataset,x_dataset_tru,y_dataset_tru

    #tradtitonal gradient of log-loss
    def grad_w(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b))*x
        return gradient

    def grad_b(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b))
        return gradient

    #calculate exceptation of truncated normal distribution E(x|c<x) 
    def E(c):
        #S(x)=1-F(x) F is cdf
        S = 1 - norm.cdf(c)
        exp = norm.pdf(c) / S
        return exp

    #truncated gradient of log-loss
    def grad_w2(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b+E(C-np.dot(theta,x))))*x
        return gradient

    def grad_b2(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b+E(C-np.dot(theta,x))))

        return gradient
        

    #main begin
    #get truncated samples
    x_dataset,y_dataset,x_dataset_tru,y_dataset_tru = tru_data_generate(N)
    '''
    #plot data sample
    plt.scatter(x_dataset,y_dataset)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    x = np.linspace(-4,4,100)
    y = theta_star[0]*x
    c = 0 * x + C
    plt.plot(x, y, color='black', linestyle='--',label=r'y = $\theta$ * x')
    plt.plot(x, c, color='black', linestyle='-',label='truncation')
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14)
    plt.legend()
    plt.show()
    '''
    #traditional sgd
    #y = theta * x + b
    theta = np.zeros(d)
    b = 0
    step = 0.001 #learning step
    t = 0
    max_iter = 500
    error=[]
    for j in range(max_iter):
        arr=np.array(range(N))
        np.random.shuffle(arr)
        #update theta,b
        for i in arr:
            theta = theta + step * grad_w(theta,b,x_dataset[i],y_dataset[i])
            b = b + step * grad_b(theta,b,x_dataset[i],y_dataset[i])
            #theta = theta/(np.linalg.norm(theta))
            t += 1
            error.append(np.sqrt((theta-theta_star)**2+b**2)/(theta_star**2))


    #truncated sgd
    #y = theta * x + b
    theta2 = np.zeros(d)
    b2 = 0
    step = 0.001 #learning step
    t = 0
    error2=[]
    for j in range(max_iter):
        arr=np.array(range(N))
        np.random.shuffle(arr)
        #update theta,b
        for i in arr:
            theta2 = theta2 + step * grad_w2(theta2,b2,x_dataset[i],y_dataset[i])
            b2 = b2 + step * grad_b2(theta2,b2,x_dataset[i],y_dataset[i])
            #theta = theta/(np.linalg.norm(theta))
            t += 1
            error2.append(np.sqrt((theta2-theta_star)**2+b2**2)/(theta_star**2))
        
    #print result of two methods
    print("C=",C)
    print("tradition SGD result:")
    print("theta1=",theta)
    print("b1=",b)
    print("error1 = ",error[-1])
    print("truncated SGD result:")
    print("theta2=",theta2)
    print("b2=",b2)
    print("error2 = ",error2[-1])
    
    #plot error - iter
    plt.scatter(range(N*max_iter),error,color='blue',s=1, label='Standard SGD regression')
    plt.scatter(range(N*max_iter),error2,color='red',s=1, label='Truncated SGD regression')
    plt.xlabel('iter',fontsize=14)
    plt.ylabel(r'error of $\theta$',fontsize=14)
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
    
    #plot datapoint
    plt.scatter(x_dataset,y_dataset)
    plt.scatter(x_dataset_tru,y_dataset_tru,marker='o',c='',edgecolors='grey')
    #set size
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    #truncated line
    x2 = np.linspace(-4,4,100)
    y2 = theta2[0]*x2+b2
    plt.plot(x2, y2, '-r', label='Truncated SGD regression')
    #traditional sgd line
    x1 = np.linspace(-4,4,100)
    y1 = theta[0]*x1+b
    plt.plot(x1, y1, '-b', label='Standard SGD regression')
    #real line
    x = np.linspace(-4,4,100)
    y = theta_star[0]*x
    c = 0 * x + C
    plt.plot(x, y, color='black', linestyle='-',label=r'y = $\theta$ * x')
    plt.plot(x, c, color='black', linestyle='--',label='truncation')
    plt.xlabel('x',fontsize=20)
    plt.ylabel('y',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.show()
    
