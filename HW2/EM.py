import numpy as np


# 模拟采样n次抛硬币的结果
def toss_result(n,s1,s2,p,q,r):
    Y = []
    for i in range(n):
        flag = np.random.rand(2)
        if flag[0] < s1:
            if flag[1] < p:
                Y.append(1)
            else:
                Y.append(0)
        if flag[0] > s1 and flag[0] < s2 +s1:
            if flag[1] < q:
                Y.append(1)
            else:
                Y.append(0)
        else:   
            if flag[1] < r:
                Y.append(1)
            else:
                Y.append(0)    
    return Y

s1,s2,p,q,r = 0.3,0.2,0.7,0.3,0.9
n = 1000
y = toss_result(n,s1,s2,p,q,r)
epoch = 10

# 定义初值
params = { 
    's1': [0.35],
    's2': [0.15],
    'p': [0.6],     
    'q': [0.4],      
    'r': [0.8],      
    'u1': np.zeros(n),      
    'u2': np.zeros(n)}    

print(0, params['s1'], params['s2'], params['p'], params['q'], params['r'])

for j in range(epoch):
    # E-step
    for i in range(n):    
        p = params['p'][0]  
        q = params['q'][0]
        r = params['r'][0]
        s1 = params['s1'][0]
        s2 = params['s2'][0]
        params['u1'][i] = (s1 * pow(p, y[i]) * pow(1-p, 1-y[i])) / (s1 * pow(p, y[i]) * pow(1-p, 1-y[i]) + s2 * pow(q, y[i]) * pow(1-q, 1-y[i]) + (1-s1-s2) * pow(r,y[i])*pow(1-r,1-y[i]))
        params['u2'][i] = (s2 * pow(q, y[i]) * pow(1-q, 1-y[i])) / (s1 * pow(p, y[i]) * pow(1-p, 1-y[i]) + s2 * pow(q, y[i]) * pow(1-q, 1-y[i]) + (1-s1-s2) * pow(r,y[i])*pow(1-r,1-y[i]))

    # M-step 
    u1 = params['u1']
    u2 = params['u2']       
    params['s1'][0] = sum(u1) / n
    params['s2'][0] = sum(u2) / n
    params['p'][0] = sum([u1[i] * y[i] for i in range(n)]) / sum(u1)
    params['q'][0] = sum([u2[i] * y[i] for i in range(n)]) / sum(u2)
    params['r'][0] = sum([(1-u1[i]-u2[i]) * y[i] for i in range(n)]) / sum([1-u1_i-u2_i for u1_i,u2_i in zip(u1,u2)])

    print(j+1, params['s1'], params['s2'], params['p'], params['q'], params['r'])