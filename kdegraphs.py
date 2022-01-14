import seaborn as sns
from numpy import array
import matplotlib.pyplot as plt


list_of_dicts1 = []
fitness1 = []
list_of_dicts2 = []
fitness2 = []


with open('infosoloartificial.txt','r') as f:
    lines = f.read().split('\n')
    lines.pop(35)
    for i in range(len(lines)):
        list_of_dicts1.append(eval(lines[i]))

with open('infotrainartificialtesttesis.txt','r') as f:
    lines = f.read().split('\n')
    lines.pop(35)
    for i in range(len(lines)):
        list_of_dicts2.append(eval(lines[i]))

for k in range(1,32):
    for i in range(35): #repeticion max 35
        for j in range(k-1,k): #iteracion
            fitness1 += list_of_dicts1[i][j]['Fitness']

    for i in range(35): #repeticion max 35
        for j in range(k-1,k): #iteracion
            fitness2 += list_of_dicts2[i][j]['Fitness']


    plt.clf()
    sns.kdeplot(data=fitness1,shade=True,bw_adjust=0.85,label='synthetic-synthetic_eval')
    sns.kdeplot(data=fitness2,shade=True,bw_adjust=0.85,label='synthetic-original')

    plt.title('RMSE distribution. Iteration:{}'.format(k-1))
    plt.xlabel('RMSE')
    if k < 4:
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='best')
    plt.savefig('iteraciones(31)/{}'.format(k-1))
