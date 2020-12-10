#EECS545 Team project truncated linear regression
#real data cases
#and https://www.kaggle.com/mustafaali96/weight-height

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd



C=100# Truncation threshold C
for C in [160,170,180,190,200]:
    #N=len(x_origin_data)# number of observed data
    N=100
    step_theta = 0.0001 #learning step
    step_b = 0.01
    step_theta2 = 0.0001 #learning step
    step_b2 = 0.01
    d=1#dimension of data
    K=15 # e~N(0,K^2)
    max_iter = 3000
    '''
    theta_star=5.79130265
    b_star=-213.89597604
    '''
    theta_star=5
    b_star=-200
    theta_star=np.array([1/np.sqrt(d)]*d)
    np.random.seed(1)
    #print("theta_star=",theta_star)

    #import data from kaggle
    data = pd.read_csv('weight-height.csv')
    x_origin_data = data['Height'].values.tolist()[:N]
    y_origin_data = data['Weight'].values.tolist()[:N]

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
            x = x_origin_data[num]
            y = y_origin_data[num]
            
            #truncate
            if phi(y):
                x_dataset.append(x)
                y_dataset.append(y)
            else:
                x_dataset_tru.append(x)
                y_dataset_tru.append(y)
            num += 1
                
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
        exp = norm.pdf(c) / (S+0.0001)
        return exp

    #truncated gradient of log-loss
    def grad_w2(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b+K* E((C-np.dot(theta,x)-b)/K) ))*x
        '''
        print(x,y)
        print("r1=",(y-(np.dot(theta,x)+b+K* E((C-np.dot(theta,x)-b)/K) )))
        print("r2=",K*E((C-np.dot(theta,x)-b)/K))
        '''
        return gradient

    def grad_b2(theta,b,x,y):
        gradient=(y-(np.dot(theta,x)+b+K*E((C-np.dot(theta,x)-b)/K)))
        return gradient

    #calculate loss=(y-(wx+b))^2
    def loss_f(x,y,w,b):
        sum = 0
        for i in range(len(x)):
            sum += (y[i]-(w[0] * x[i] +b))**2
        return sum / len(x)
        

    #main begin
    #get truncated samples
    x_dataset,y_dataset,x_dataset_tru,y_dataset_tru = tru_data_generate(N)

    '''
    #plot data sample
    plt.scatter(x_dataset,y_dataset)
    x = np.linspace(60,80,100)
    c = 0 * x + C
    plt.plot(x, c, color='black', linestyle='-',label='truncation')
    plt.scatter(x_dataset_tru,y_dataset_tru,marker='o',c='',edgecolors='grey')
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14)
    plt.legend()
    plt.show()
    '''

    #traditional sgd
    #y = theta * x + b
    theta = np.zeros(d)
    theta[0]=theta_star
    b = b_star
    t = 0
    theta_list=[]
    b_list=[]
    error = []
    for j in range(0,max_iter):
        tmp1 = j
        #print("tmp1=",tmp1)
        arr=np.array(range(len(x_dataset)))
        np.random.shuffle(arr)
        #update theta,b
        for i in arr:
            theta_tmp = theta + (step_theta / (j/100 + 1)) * grad_w(theta,b,x_dataset[i],y_dataset[i])
            b_tmp = b + (step_b) * grad_b(theta,b,x_dataset[i],y_dataset[i])
            theta = theta_tmp
            b = b_tmp
            #theta = theta/(np.linalg.norm(theta))
            t += 1
        #theta_list.append(theta)
        #b_list.append(b)
        error.append(np.sqrt((theta-theta_star)**2+(b-b_star)**2)/(theta_star**2+b_star**2))

    K=np.sqrt(loss_f(x_dataset,y_dataset,theta,b))

            
    #truncated sgd
    #y = theta * x + b
    theta2 = np.zeros(d)
    theta2[0]=theta_star
    b2 = b_star
    t = 0
    theta_list2=[]
    b_list2=[]
    error2 = []
    for j in range(max_iter):
        arr=np.array(range(len(x_dataset)))
        np.random.shuffle(arr)
        #update theta,b
        for i in arr:
            theta2_tmp = theta2 + (step_theta2 / (j/100+1)) * grad_w2(theta2,b2,x_dataset[i],y_dataset[i])
            b2_tmp = b2 + (step_b2) * grad_b2(theta2,b2,x_dataset[i],y_dataset[i])
            theta2 = theta2_tmp
            b2 = b2_tmp
            #theta = theta/(np.linalg.norm(theta))
            t += 1
        #theta_list2.append(theta2)
        #b_list2.append(b2)
        error2.append(np.sqrt((theta2-theta_star)**2+(b2-b_star)**2)/(theta_star**2+b_star**2))

    #truncated sgd
    #y = theta * x + b

        
    #print result of two methods
    print("C=",C)
    print("tradition SGD result:")
    print("theta1=",theta)
    print("b1=",b)
    print("error of sd sgd=",error[-1])

    print("truncated SGD result:")
    print("K=",K)
    print("theta2=",theta2)
    print("b2=",b2)
    print("error of truncated sgd=",error2[-1])


    '''
    #plot theta - iter
    plt.xlim(0, max_iter)
    #plt.ylim(-1000, 1000)
    plt.scatter(range(max_iter),theta_list,color='blue',s=1, label='Standard SGD regression')
    #plt.scatter(range(N*max_iter),loss2,color='red',s=1, label='Truncated SGD regression')
    plt.xlabel('iter',fontsize=14)
    plt.ylabel(r'$\theta$',fontsize=14)
    plt.legend()
    plt.show()

    #plot b - iter
    plt.xlim(0, max_iter)
    #plt.ylim(-1000, 1000)
    plt.scatter(range(max_iter),b_list,color='blue',s=1, label='Standard SGD regression')
    #plt.scatter(range(N*max_iter),loss2,color='red',s=1, label='Truncated SGD regression')
    plt.xlabel('iter',fontsize=14)
    plt.ylabel(r'b',fontsize=14)
    plt.legend()
    plt.show()
    '''

    #plot loss - iter
    plt.scatter(range(max_iter),error,color='blue',s=1, label='Standard SGD regression')
    plt.scatter(range(max_iter),error2,color='red',s=1, label='truncate SGD regression')
    plt.xlabel('iter',fontsize=14)
    plt.ylabel(r'loss',fontsize=14)
    plt.legend()
    plt.show()


    #plot datapoint
    plt.scatter(x_dataset,y_dataset)
    plt.scatter(x_dataset_tru,y_dataset_tru,marker='o',c='',edgecolors='grey')
    #set size
    #plt.xlim(18, 65)
    #plt.ylim(-1000, 65000)
    #truncated line

    #traditional sgd line
    x1 = np.linspace(60,80,100)
    y1 = theta[0]*x1+b
    plt.plot(x1, y1, '-b', label='Standard SGD regression')
    #turncated sgd line
    x2 = np.linspace(60,80,100)
    y2 = theta2[0]*x2+b2
    plt.plot(x2, y2, '-r', label='Truncated SGD regression')
    #truncated line
    c= 0*x2+C
    #plt.plot(x, y, color='black', linestyle='-',label=r'y = $\theta$ * x')
    plt.plot(x2, c, color='black', linestyle='--',label='truncation')
    plt.xlabel('Height(inch)',fontsize=14)
    plt.ylabel('Weight(pound)',fontsize=14)
    plt.legend()
    plt.show()


