import pandas as pd
import numpy as np

num_registros = 500
columnas = ['EDAD', 'AF', 'AHF CA','MENARCA','HIJO']
path = '/home/joel/Desktop/Experimento_RegresionPso/Dataset_artificialtest.csv'

def codify_af (row):
    if row['AF'] == 3 :
        return 2
    else:
        return 1


df = pd.DataFrame(index=range(1,num_registros+1),columns=columnas)
df['EDAD'] = np.round((85-35)*np.random.rand(num_registros,1) + 35)
df['MENARCA'] = np.round(np.random.normal(11.6,1.1,(num_registros,1)))
df['AF'] = np.round(2*np.random.rand(num_registros,1)+1)
df['AHF CA'] = df.apply (lambda row: codify_af(row), axis=1)
df['HIJO'] = np.random.choice([0,1], (num_registros,1),p=[0.25,0.75])

#print(df)
df.to_csv(path, index=False)
