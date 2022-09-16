import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

z_scaler = preprocessing.StandardScaler()

def cut_sequence_error( seq_i, word_length, stride ):
  all_seqs = []
  all_labs = []
  i = 0
  while (i < len(seq_i)-word_length ):
    if ( i + word_length <= len(seq_i)-1 ):
      all_seqs.append( seq_i[i : i + word_length] )
      all_labs.append( [seq_i[i + word_length]] )
    else:
      dif_vals = i + word_length - len(seq_i)
      final_list = seq_i[i : i + word_length] + [0]*dif_vals
      all_seqs.append( final_list )
      all_labs.append( [0] )
    i = i + stride
  return all_seqs, all_labs

def get_sequences_matrix_error( train_seqs, word_window, word_shift ):
  train_seqs_expand = []
  train_labels_expand = []
  for i, each_seq in enumerate( train_seqs ):
    new_seq, new_lab = cut_sequence_error( train_seqs[i], word_window, word_shift )
    train_seqs_expand.extend( new_seq )
    train_labels_expand.extend( new_lab )
  return train_seqs_expand, train_labels_expand

def splitdata(df):    
    split = train_test_split(df[1], df[2], test_size=0.25, random_state=42)
    (trainX, testX, trainY, testY) = split
    trainX=np.expand_dims(trainX,axis=-1)
    testX=np.expand_dims(testX,axis=-1)
    trainY=np.expand_dims(trainY,axis=-1)
    testY=np.expand_dims(testY,axis=-1)
    return trainX, testX, trainY, testY

def funcion(data, col_lugar, col_fecha):    
    index,X,Y,index_total=[],[],[],[]
    for i in range(len(data[col_lugar].unique())):
        index.append(i)
    
    bh1 = pd.pivot_table(data, columns=[col_fecha],index=[col_lugar] )
    bh2 = bh1.reset_index()

    for i in bh2[col_lugar].unique():
        df5=(bh2.loc[bh2[col_lugar]==i])
        del df5[col_lugar]
        df5=df5.to_numpy()
        a = get_sequences_matrix_error(df5,6,2) #6 2
        X.append(np.array(a[0]))
        Y.append(np.array(a[1]))
    
    n_repetir = len(a[0]) 

    for v in index:
        l = [v] * n_repetir
        index_total = index_total + l
    X=np.concatenate(X,axis=0)
    Y=np.concatenate(Y,axis=0)

    index_total1 = np.reshape(index_total, (len(index_total),1)) 
    total5=np.concatenate((index_total1,X,Y), axis=1)
    total1=total5[~np.isnan(total5).any(axis=1)]
    
    ind,dX,dY,a,muestra=[],[],[],total1.shape[1],[]
   
    # 

    for i in total1:
        ind.append(np.array(i[:1]))
        dX.append(np.array(i[1:a-1]))
        dY.append(np.array(i[a-1:]))
        muestra.append(np.array(i[1:a]))
    
    split = train_test_split(dX,dY, test_size=0.2, random_state=42)
    (x_train, x_test, y_train, y_test) = split
    
    train_split = np.concatenate((x_train,y_train), axis=1)
    
    scaler = StandardScaler()
    scaler.fit(train_split)
    X=scaler.transform(muestra)
    
    DX,DY,b=[],[],X.shape[1]
    for i in X:
        #ind.append(np.array(i[:1]))
        DX.append(np.array(i[0:b-1]))
        DY.append(np.array(i[b-1:]))
        #muestra.append(np.array(i[1:a]))
    
    return np.array(ind),np.array(DX),np.array(DY), scaler