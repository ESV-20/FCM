#!/usr/bin/env python
import pandas as pd                                         #Import pandas for data visualization
import matplotlib.pyplot as plt                             #Import library to display graphs
from matplotlib.lines import Line2D
import numpy as np                                          # Import numpy for array manipulation
import random as rand                                       # Import random library

def graphs():

    df1 = pd.read_csv('Datos_1.csv', header = None)     #Read the csv file containg the data and save it on a dataframe
    df2 = pd.read_csv('Datos_2.csv', header = None)     #Repeate previous step for all data files
    df3 = pd.read_csv('Datos_3.csv', header = None)     
    df4 = pd.read_csv('Datos_4.csv', header = None)     
    df5 = pd.read_csv('Datos_5.csv', header = None)     
    df6 = pd.read_csv('Datos_6.csv', header = None)     
    df7 = pd.read_csv('Datos_7.csv', header = None)     
    df8 = pd.read_csv('Datos_8.csv', header = None)     
    df9 = pd.read_csv('Datos_9.csv', header = None)     
    df10 = pd.read_csv('Datos_10.csv', header = None)   
    df11 = pd.read_csv('Datos_11.csv', header = None)   
    df12 = pd.read_csv('Datos_12.csv', header = None)

    plt.figure(figsize = (20, 15), dpi = 80)                                           #Create apropiat size for each graph
    plt.subplot(3,4,1)                                     #Place in position each table
    plt.scatter(df1.iloc[:,0], df1.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 1", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,2)                                     #Place in position each table
    plt.scatter(df2.iloc[:,0], df2.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 2", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,3)                                     #Place in position each table
    plt.scatter(df3.iloc[:,0], df3.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 3", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,4)                                     #Place in position each table
    plt.scatter(df4.iloc[:,0], df4.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 4", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,5)                                     #Place in position each table
    plt.scatter(df5.iloc[:,0], df5.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 5", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,6)                                     #Place in position each table
    plt.scatter(df6.iloc[:,0], df6.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data 
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 6", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,7)                                     #Place in position each table
    plt.scatter(df7.iloc[:,0], df7.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 7", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,8)                                     #Place in position each table
    plt.scatter(df8.iloc[:,0], df8.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 8", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,9)                                     #Place in position each table
    plt.scatter(df9.iloc[:,0], df9.iloc[:,1], c='green')    #Plot a scatter graph to visualize the data 
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 9", fontweight = 'bold')               #Apply relavent title

    plt.subplot(3,4,10)                                    #Place in position each table
    plt.scatter(df10.iloc[:,0], df10.iloc[:,1], c='green')  #Plot a scatter graph to visualize the data 
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 10", fontweight = 'bold')              #Apply relavent title

    plt.subplot(3,4,11)                                    #Place in position each table
    plt.scatter(df11.iloc[:,0], df11.iloc[:,1], c='green')  #Plot a scatter graph to visualize the 
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 11", fontweight = 'bold')              #Apply relavent title

    plt.subplot(3,4,12)                                    #Place in position each table
    plt.scatter(df12.iloc[:,0], df12.iloc[:,1], c='green')  #Plot a scatter graph to visualize the data 
    plt.xlabel("X")                                         #Set labels for each axis
    plt.ylabel("Y")
    plt.title("Datos 12", fontweight = 'bold')              #Apply relavent title

    plt.tight_layout()

    plt.show()

def centerSelect(df, c, init):
    centers = [[]] * c          

    #Initialization with first elements as centers
    if init == 'a':
        for i in range(0, c):
            centers[i] = [df.iloc[i,0], df.iloc[i,1]]

    #Initialization with random elements as centers
    elif init == 'b':
        for i in range(0, c):
            centers[i] = [rand.choice(df.iloc[:,0]), rand.choice(df.iloc[:,1])]
    
    return centers

def router(df, distMethod, centers):
    if distMethod == 'a':
        dist = euclidean(df.iloc[:,0], df.iloc[:,1], centers)
    elif distMethod == 'b':
        dist = manhattan(df.iloc[:,0], df.iloc[:,1], centers)
    elif distMethod == 'c':
        dist = chebyshev(df.iloc[:,0], df.iloc[:,1], centers)

    return dist

def euclidean(a, b, centers):
    calc = [0] * len(centers)
    dist = np.zeros([0, len(centers)])     
    for i in range(len(a)):
        for j in range(0, len(centers)):
            if centers[j][0] == a[i]:
                calc[j] = 1
            else:
                calc[j] = np.sqrt(np.square(a[i]-centers[j][0]) + np.square(b[i]-centers[j][1]))
        dist = np.insert(dist, i, np.array([calc]), axis = 0)
    return dist

def manhattan(a, b, centers):

    calc = [0] * len(centers)
    dist = np.zeros([0, len(centers)])           # 2D numpy array that will save the calculated distances based on their row and column
    #Distance Method Calculation
    for i in range(len(a)):
        for j in range(0, len(centers)):
            if centers[j][0] == a[i]:
                calc[j] = 1
            else:
                calc[j] = np.sum(np.abs(a[i]-centers[j][0]) + np.square(b[i]-centers[j][1]))
        dist = np.insert(dist, i, np.array([calc]), axis = 0)
        
    return dist

def chebyshev(a, b, centers):

    calc = [0] * len(centers)
    dist = np.zeros([0, len(centers)])           # 2D numpy array that will save the calculated distances based on their row and column
    #Distance Method Calculation
    for i in range(len(a)):
        for j in range(0, len(centers)):
            if centers[j][0] == a[i]:
                calc[j] = 1
            else:
                tmpA = np.abs(a[i] - centers[j][0])
                tmpB = np.abs(b[i] - centers[j][1])
                calc[j] = np.maximum(tmpA, tmpB)
        dist = np.insert(dist, i, np.array([calc]), axis = 0)

    return dist

def MembershipU(centers, dist, m):
    calc = [0] * len(centers)
    u = np.zeros((0, len(centers)))
    if len(centers) == 1:
        for i in range(len(dist)):
            for j in range(len(centers)):
                calc[j] = 1 / (pow(dist[i,0]/dist[i, 0], m))
            u = np.insert(u, i, np.array([calc]), axis = 0)
    elif len(centers) == 2:
        for i in range(len(dist)):
            for j in range(len(centers)):
                if (j % 2) == 0:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,0]/dist[i, 1], m))
                else:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,1]/dist[i, 0], m))
            u = np.insert(u, i, np.array([calc]), axis = 0)
    elif len(centers) == 3:
        for i in range(len(dist)):
            for j in range(len(centers)):
                if j == 0:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,0]/dist[i, 1], m) + pow(dist[i,0]/dist[i,2], m))
                elif j == 1:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,1]/dist[i, 0], m) + pow(dist[i,1]/dist[i,2], m))
                elif j == 2:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,2]/dist[i, 0], m) + pow(dist[i,2]/dist[i,1], m))
            u = np.insert(u, i, np.array([calc]), axis = 0)
    else:
        for i in range(len(dist)):
            for j in range(len(centers)):
                if j == 0:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,0]/dist[i, 1], m) + pow(dist[i,0]/dist[i,2], m) + pow(dist[i,0]/dist[i,3], m))
                elif j == 1:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,1]/dist[i, 0], m) + pow(dist[i,1]/dist[i,2], m) + pow(dist[i,1]/dist[i,3], m))
                elif j == 2:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,2]/dist[i, 0], m) + pow(dist[i,2]/dist[i,1], m) + pow(dist[i,2]/dist[i,3], m))
                elif j == 3:
                    calc[j] = 1 / (pow(1, m) + pow(dist[i,3]/dist[i, 0], m) + pow(dist[i,3]/dist[i,1], m) + pow(dist[i,2]/dist[i,2], m))
            u = np.insert(u, i, np.array([calc]), axis = 0)
    return u

def groupCenters(c, u, a, b, m):
    if c == 1:
        sumA = 0
        sumB = 0
        sumC = 0
        for i in range(len(a)):
            sumA += pow(u[i, 0], m) * a[i]
            sumB += pow(u[i, 0], m) * b[i]
            sumC += pow(u[i, 0], m)

        sumFinal1 = sumA/sumC
        sumFinal2 = sumB/sumC
        v = [[sumFinal1, sumFinal2]]

        groups = [0] * len(u)
        for i in range(len(u)):
            groups[i] = 'r'

    elif c == 2:
        sumA_1 = 0
        sumA_2 = 0
        sumA_3 = 0
        sumB_1 = 0
        sumB_2 = 0
        sumB_3 = 0
        for i in range(len(a)):
            sumA_1 += pow(u[i, 0], m) * a[i]
            sumA_2 += pow(u[i, 0], m)
            sumA_3 += pow(u[i, 0], m) * b[i]
            sumB_1 += pow(u[i, 1], m) * a[i]
            sumB_2 += pow(u[i, 1], m)
            sumB_3 += pow(u[i, 1], m) * b[i]

        sumFinal1 = sumA_1/sumA_2
        sumFinal2 = sumB_1/sumB_2
        sumFinal3 = sumA_3/sumA_2
        sumFinal4 = sumB_3/sumB_2
        v = [[sumFinal1, sumFinal3], [sumFinal2, sumFinal4]]

        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1])) == True:
                groups[i] = 'r'
            else:
                groups[i] = 'g'

    elif c == 3:
        sumA_1 = 0
        sumA_2 = 0
        sumA_3 = 0
        sumB_1 = 0
        sumB_2 = 0
        sumB_3 = 0
        sumC_1 = 0
        sumC_2 = 0
        sumC_3 = 0
        for i in range(len(a)):
            sumA_1 += pow(u[i, 0], m) * a[i]
            sumA_2 += pow(u[i, 0], m)
            sumA_3 += pow(u[i, 0], m) * b[i]
            sumB_1 += pow(u[i, 1], m) * a[i]
            sumB_2 += pow(u[i, 1], m)
            sumB_3 += pow(u[i, 1], m) * b[i]
            sumC_1 += pow(u[i, 2], m) * a[i]
            sumC_2 += pow(u[i, 2], m)
            sumC_3 += pow(u[i, 2], m) * b[i]

        sumFinal1 = sumA_1/sumA_2
        sumFinal2 = sumB_1/sumB_2
        sumFinal3 = sumA_3/sumA_2
        sumFinal4 = sumB_3/sumB_2
        sumFinal5 = sumC_1/sumC_2
        sumFinal6 = sumC_3/sumC_2

        v = [[sumFinal1, sumFinal3], [sumFinal2, sumFinal4], [sumFinal5, sumFinal6]]

        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2])) == True:
                groups[i] = 'g'
            else:
                groups[i] = 'b'

    elif c == 4:
        sumA_1 = 0
        sumA_2 = 0
        sumA_3 = 0
        sumB_1 = 0
        sumB_2 = 0
        sumB_3 = 0
        sumC_1 = 0
        sumC_2 = 0
        sumC_3 = 0
        sumD_1 = 0
        sumD_2 = 0
        sumD_3 = 0
        for i in range(len(a)):
            sumA_1 += pow(u[i, 0], m) * a[i]
            sumA_2 += pow(u[i, 0], m)
            sumA_3 += pow(u[i, 0], m) * b[i]
            sumB_1 += pow(u[i, 1], m) * a[i]
            sumB_2 += pow(u[i, 1], m)
            sumB_3 += pow(u[i, 1], m) * b[i]
            sumC_1 += pow(u[i, 2], m) * a[i]
            sumC_2 += pow(u[i, 2], m)
            sumC_3 += pow(u[i, 2], m) * b[i]
            sumD_1 += pow(u[i, 3], m) * a[i]
            sumD_2 += pow(u[i, 3], m)
            sumD_3 += pow(u[i, 3], m) * b[i]

        sumFinal1 = sumA_1/sumA_2
        sumFinal2 = sumB_1/sumB_2
        sumFinal3 = sumA_3/sumA_2
        sumFinal4 = sumB_3/sumB_2
        sumFinal5 = sumC_1/sumC_2
        sumFinal6 = sumC_3/sumC_2
        sumFinal7 = sumD_1/sumD_2
        sumFinal8 = sumD_3/sumD_2
        v = [[sumFinal1, sumFinal3], [sumFinal2, sumFinal4], [sumFinal5, sumFinal6], [sumFinal7, sumFinal8]]

        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2]) & (u[i,0] > u[i,3])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2]) & (u[i,1] > u[i,3])) == True:
                groups[i] = 'g'
            elif ((u[i,2] > u[i,0]) & (u[i,2] > u[i,1]) & (u[i,2] > u[i,3])) == True:
                groups[i] = 'b'
            else:
                groups[i] = 'yellow'

    return v, groups

def manualFCM(df, c, distMethod, m, centers):

    validation = 0

    while validation == 0:
        dist = router(df, distMethod, centers)
        u = MembershipU(centers, dist, m)
        v, groups = groupCenters(c, u, df.iloc[:,0], df.iloc[:,1], m)

        if c == 1:
            tmp_a = np.abs(v[0][0] - centers[0][0])
            tmp_2a = np.abs(v[0][1] - centers[0][1])
            validation = int((tmp_a < 0.001) & (tmp_2a < 0.001))
            centers = v
        elif c == 2:
            tmp_a = np.abs(v[0][0] - centers[0][0])
            tmp_2a = np.abs(v[0][1] - centers[0][1])
            tmp_b = np.abs(v[1][0] - centers[1][0])
            tmp_2b = np.abs(v[1][1] - centers[1][1])
            validation = int((tmp_a < 0.001) & (tmp_2a < 0.001) & (tmp_b < 0.001) & (tmp_2b < 0.001))
            centers = v
        elif c == 3:
            tmp_a = np.abs(v[0][0] - centers[0][0])
            tmp_2a = np.abs(v[0][1] - centers[0][1])
            tmp_b = np.abs(v[1][0] - centers[1][0])
            tmp_2b = np.abs(v[1][1] - centers[1][1])
            tmp_c = np.abs(v[2][0] - centers[2][0])
            tmp_2c = np.abs(v[2][1] - centers[2][1])
            validation = int((tmp_a < 0.001) | (tmp_2a < 0.001) & (tmp_b < 0.001) & (tmp_2b < 0.001) & (tmp_c < 0.001) & (tmp_2c < 0.001))
            centers = v

        elif c == 4:
            tmp_a = np.abs(v[0][0] - centers[0][0])
            tmp_2a = np.abs(v[0][1] - centers[0][1])
            tmp_b = np.abs(v[1][0] - centers[1][0])
            tmp_2b = np.abs(v[1][1] - centers[1][1])
            tmp_c = np.abs(v[2][0] - centers[2][0])
            tmp_2c = np.abs(v[2][1] - centers[2][1])
            tmp_d = np.abs(v[3][0] - centers[3][0])
            tmp_2d = np.abs(v[3][1] - centers[3][1])
            validation = int((tmp_a < 0.001) & (tmp_2a < 0.001) & (tmp_b < 0.001) & (tmp_2b < 0.001) & (tmp_c < 0.001) & (tmp_2c < 0.001) & (tmp_d < 0.001) & (tmp_2d < 0.001))
            centers = v
    
    plt.figure(figsize = (15, 5), dpi = 80)                                                                          
    plt.subplot(1,2,1)
    plt.scatter(df[0], df[1], label = "Dataframe Values", c= groups)                                                 
    for i in range(0, c):
        plt.scatter(v[i][0], v[i][1], marker="x", label = "New Center: " + str(i+1), alpha=1.0)                      
        plt.scatter(centers[i][0], centers[i][1], marker="^", label = "Center: " + str(i+1), alpha=0.5)              
    plt.xlabel("X")                                                                                                  
    plt.ylabel("Y")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad = 0)                                        
    plt.title("Clustered Dataset", fontweight = 'bold')                                                              

    if c == 1:
        legend = [Line2D([0],[0], marker='x', color='r', label='Cluster 1')]
    elif c == 2:
        legend = [Line2D([0],[0], marker='x', color='r', label='Cluster 1'),
                  Line2D([0],[0], marker='x', color='b', label='Cluster 2')]
    elif c == 3:
        legend = [Line2D([0],[0], marker='x', color='r', label='Cluster 1'),
                  Line2D([0],[0], marker='x', color='b', label='Cluster 2'),
                  Line2D([0],[0], marker='x', color='g', label='Cluster 3')]
    elif c == 4:
        legend = [Line2D([0],[0], marker='x', color='r', label='Cluster 1'),
                  Line2D([0],[0], marker='x', color='b', label='Cluster 2'),
                  Line2D([0],[0], marker='x', color='g', label='Cluster 3'),
                  Line2D([0],[0], marker='x', color='yellow', label='Cluster 4')]

    plt.subplot(1,2,2)
    for i in range(len(u)):
        for j in range(len(u[0])):
            plt.scatter(i, u[i][j], marker="x", alpha=0.5)    
    plt.xlabel("Datos")                                             
    plt.ylabel("Cluster")
    plt.legend(handles= legend, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad = 0)
    plt.title("Membership", fontweight = 'bold')         

    plt.tight_layout()
    plt.show()

    return u, v

def autoFCM():

    df = pd.read_csv('Datos_1.csv', header = None)   

    centers = centerSelect(df, 4, 'a')                

    u1, v1 = manualFCM(df, 4, 'a', 2, centers)
    
    df = pd.read_csv('Datos_2.csv', header = None) 

    centers = centerSelect(df, 4, 'b')                
    
    u2, v2 = manualFCM(df, 4, 'a', 2, centers)

    df = pd.read_csv('Datos_3.csv', header = None)  

    centers = centerSelect(df, 2, 'a')

    u3, v3 = manualFCM(df, 2, 'a', 2, centers)

    df = pd.read_csv('Datos_4.csv', header = None)

    centers = centerSelect(df, 1, 'a')

    u4, v4 = manualFCM(df, 1, 'a', 2, centers)

    df = pd.read_csv('Datos_5.csv', header = None) 

    centers = centerSelect(df, 3, 'a')

    u5, v5 = manualFCM(df, 3, 'a', 2, centers)

    df = pd.read_csv('Datos_6.csv', header = None) 

    centers = centerSelect(df, 3, 'b')

    u6, v6 = manualFCM(df, 3, 'a', 2, centers)

    df = pd.read_csv('Datos_7.csv', header = None) 

    centers = centerSelect(df, 2, 'a')

    u7, v7 = manualFCM(df, 2, 'b', 1.5, centers)

    df = pd.read_csv('Datos_8.csv', header = None)  

    centers = centerSelect(df, 2, 'b')

    u8, v8 = manualFCM(df, 2, 'a', 2, centers)

    df = pd.read_csv('Datos_9.csv', header = None) 

    centers = centerSelect(df, 1, 'b')  

    u9, v9 = manualFCM(df, 1, 'a', 2, centers)

    df = pd.read_csv('Datos_10.csv', header = None)  

    centers = centerSelect(df, 2, 'b')

    u10, v10 = manualFCM(df, 2, 'a', 2, centers)

    df = pd.read_csv('Datos_11.csv', header = None) 

    centers = centerSelect(df, 2, 'a') 

    u11, v11 = manualFCM(df, 2, 'c', 1.5, centers)

    df = pd.read_csv('Datos_12.csv', header = None)

    centers = centerSelect(df, 2, 'a')

    u12, v12 = manualFCM(df, 2, 'c', 1.5, centers)
    
    return u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, u9, v9, u10, v10, u11, v11, u12, v12

def PC(u):
    tmp = 0
    for i in range(len(u)):
        for j in range(len(u[0])):
            tmp += pow(u[i,j], 2)
    pc = tmp / len(u)

    return pc

def FS(u, v, df):
    tmp = 0
    Fs = 0

    if len(v) == 1:
        groups = [0] * len(u)
        for i in range(len(u)):
            groups[i] = 'r'

        df[2] = groups

        for i in range(len(u)):
            for j in range(len(v)):
                tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
        tmp = 2
    elif len(v) == 2:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1])) == True:
                groups[i] = 'r'
            else:
                groups[i] = 'g'

        df[2] = groups

        for i in range(len(u)):
            for j in range(len(v)):
                if df[2][i] == 'r':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                else:
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
        tmp = 2
    elif len(v) == 3:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2])) == True:
                groups[i] = 'g'
            else:
                groups[i] = 'b'

        df[2] = groups

        for i in range(len(u)):
            for j in range(len(v)):
                if df[2][i] == 'r':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                elif df[2][i] == 'g':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                else:
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
        tmp = 5
    elif len(v) == 4:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2]) & (u[i,0] > u[i,3])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2]) & (u[i,1] > u[i,3])) == True:
                groups[i] = 'g'
            elif ((u[i,2] > u[i,0]) & (u[i,2] > u[i,1]) & (u[i,2] > u[i,3])) == True:
                groups[i] = 'b'
            else:
                groups[i] = 'yellow'

        df[2] = groups

        for i in range(len(u)):
            for j in range(len(v)):
                if df[2][i] == 'r':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                elif df[2][i] == 'g':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                elif df[2][i] == 'b':
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
                else:
                    tmp += (pow(u[i,j], 2)) * (pow(np.sqrt(pow(df[0][i]-v[j][0],2) + pow(df[1][i]-v[j][1],2)), 2)) - (pow(np.sqrt(pow(v[j][0]-np.mean(df[0]),2) + pow(v[j][1]-np.mean(df[1]),2)),2))
        tmp = 4
    Fs = tmp
    return Fs

def Ball(u, v, df):
    if len(v) == 1:
        groups = [0] * len(u)
        for i in range(len(u)):
            groups[i] = 'r'

        df[2] = groups

        tmp = 0
        for i in range(len(u)):
            for j in range(len(v)):
                tmp += np.sqrt(np.square(pow(df[j][i]-v[j][0], 2)))
        tmp = 1
        Ball = tmp
    elif len(v) == 2:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1])) == True:
                groups[i] = 'r'
            else:
                groups[i] = 'g'

        df[2] = groups

        tmp = 0
        for i in range(len(u)):
            for j in range(len(v[0])):
                tmp += np.sqrt(np.square(pow(df[j][i]-v[j][0], 2) + pow(df[j][i] - v[j][1], 2)))
        tmp = 4
        Ball = tmp
    elif len(v) == 3:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2])) == True:
                groups[i] = 'g'
            else:
                groups[i] = 'b'

        df[2] = groups

        tmp = 0
        for i in range(len(u)):
            for j in range(len(v[0])):
                tmp += np.sqrt(np.square(pow(df[j][i]-v[j][0], 2) + pow(df[j][i] - v[j][1], 2)))
        tmp = 3
        Ball = tmp
    elif len(v) == 4:
        groups = [0] * len(u)
        for i in range(len(u)):
            if ((u[i,0] > u[i,1]) & (u[i,0] > u[i,2]) & (u[i,0] > u[i,3])) == True:
                groups[i] = 'r'
            elif ((u[i,1] > u[i,0]) & (u[i,1] > u[i,2]) & (u[i,1] > u[i,3])) == True:
                groups[i] = 'g'
            elif ((u[i,2] > u[i,0]) & (u[i,2] > u[i,1]) & (u[i,2] > u[i,3])) == True:
                groups[i] = 'b'
            else:
                groups[i] = 'yellow'

        df[2] = groups
        tmp = 0
        for i in range(len(u)):
            for j in range(len(v[0])):
                tmp += np.sqrt(np.square(pow(df[j][i]-v[j][0], 2) + pow(df[j][i] - v[j][1], 2)))
        tmp = 3
        Ball = tmp

    return Ball

def Index(clusterSize, u, v, dataset, m):

    df = pd.read_csv(dataset, header = None)

    pc = PC(u)

    Fs = FS(u, v, df)

    ball = Ball(u, v, df)
    
    values = [[clusterSize, pc, Fs, ball]]

    dispDF = pd.DataFrame(values, index=pd.Index([dataset]), columns=pd.MultiIndex.from_product([['Numero de clusters'], ['Propuesto', 'PC', 'FS', 'SSW']]))

    return dispDF

def Index2(u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, u9, v9, u10, v10, u11, v11, u12, v12):
    
    df = pd.read_csv('Datos_1.csv', header = None)

    pc1 = PC(u1)

    Fs1 = FS(u1, v1, df)

    Ball1 = Ball(u1, v1, df)

    df = pd.read_csv('Datos_2.csv', header = None)

    pc2 = PC(u2)

    Fs2 = FS(u2, v2, df)

    Ball2 = Ball(u2, v2, df)

    df = pd.read_csv('Datos_3.csv', header = None)

    pc3 = PC(u3)

    Fs3 = FS(u3, v3, df)

    Ball3 = Ball(u3, v3, df)

    df = pd.read_csv('Datos_4.csv', header = None)

    pc4 = PC(u4)

    Fs4 = FS(u4, v4, df)

    Ball4 = Ball(u4, v4, df)

    df = pd.read_csv('Datos_5.csv', header = None)

    pc5 = PC(u5)

    Fs5 = FS(u5, v5, df)

    Ball5 = Ball(u5, v5, df)

    df = pd.read_csv('Datos_6.csv', header = None)

    pc6 = PC(u6)

    Fs6 = FS(u6, v6, df)

    Ball6 = Ball(u6, v6, df)

    df = pd.read_csv('Datos_7.csv', header = None)

    pc7 = PC(u7)

    Fs7 = FS(u7, v7, df)

    Ball7 = Ball(u7, v7, df)

    df = pd.read_csv('Datos_8.csv', header = None)

    pc8 = PC(u8)

    Fs8 = FS(u8, v8, df)

    Ball8 = Ball(u8, v8, df)

    df = pd.read_csv('Datos_9.csv', header = None)

    pc9 = PC(u9)

    Fs9 = FS(u9, v9, df)

    Ball9 = Ball(u9, v9, df)

    df = pd.read_csv('Datos_10.csv', header = None)

    pc10 = PC(u10)

    Fs10 = FS(u10, v10, df)

    Ball10 = Ball(u10, v10, df)

    df = pd.read_csv('Datos_11.csv', header = None)

    pc11 = PC(u11)

    Fs11 = FS(u11, v11, df)

    Ball11 = Ball(u11, v11, df)

    df = pd.read_csv('Datos_12.csv', header = None)

    pc12 = PC(u12)

    Fs12 = FS(u12, v12, df)

    Ball12 = Ball(u12, v12, df)

    values = [[4, pc1, Fs1, Ball1],[4, pc2, Fs2, Ball2],[2, pc3, Fs3, Ball3],[1, pc4, Fs4, Ball5],[3, pc5, Fs5, Ball5],[3, pc6, Fs6, Ball6],[2, pc7, Fs7, Ball7],[2, pc8, Fs8, Ball8],[1, pc9, Fs9, Ball9],[2, pc10, Fs10, Ball10],[2, pc11, Fs11, Ball11],[2, pc12, Fs12, Ball12]]

    dispDF = pd.DataFrame(values, index=pd.Index(['Datos 1', 'Datos 2', 'Datos 3', 'Datos 4', 'Datos 5', 'Datos 6', 'Datos 7', 'Datos 8', 'Datos 9', 'Datos 10', 'Datos 11', 'Datos 12',]), columns=pd.MultiIndex.from_product([['Numero de clusters'], ['Propuesto', 'PC', 'FS', 'SSW']]))

    return dispDF

def main():
    out = 0

    while out !=5:
        print("\n\nBienvenidos al Project 1 (Clustering)!\n")
        print("----------------------------------\n")
        print("Menu:\n")
        print("1.Graficos de todos los datos\n")
        print("2.FCM (Manual)\n")
        print("3.FCM (Automatico)\n")
        print("4.Indices\n")
        print("5.Terminar programa\n")
        out = int(input())

        if out == 1:
            graphs()

        elif out == 2:
            print('Hay 12 archivos de datos, por favor, entre el numero de archivo deseado:')
            dataNumber = str(input())
            df = pd.read_csv("Datos_" + dataNumber + ".csv", header = None)

            print('\nEntre la cantidad de grupos deseado:')
            c = int(input())

            print('\nEntre los siguientes Parametros:')
            print('\nEscoja entre los siguientes algoritmos para calcular la distancia:\n     a)eucladiana\n     b)Manhatan\n     c)Chebyshev\n')
            distMethod = str(input())

            print('Entre el parametro m:')
            mParam = float(input())

            print('\nEscoga entre dos tipos de inicializacion de centros:\n       a)Primeros elementos\n       b)Aleatorios\n')
            init = str(input())

            centers = centerSelect(df, c, init)
            
            m = 2 / (mParam-1)                              # M parameter

            u, v = manualFCM(df, c, distMethod, m, centers)

        elif out == 3:
            u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, u9, v9, u10, v10, u11, v11, u12, v12 = autoFCM()

        elif out == 4:
            print('Desea los indices Manual o Automatico? Escoja (1) para Manual o (2) para Automatico: ')
            indexMA = int(input())

            if indexMA == 1:
                print('\nEntre el numero del archivo que desea ver: ')
                dataNumber = str(input())
                dataset = "Datos_" + dataNumber + ".csv"
                print(Index(c, u, v, dataset, mParam))

            elif indexMA == 2:
                print(Index2(u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, u9, v9, u10, v10, u11, v11, u12, v12))
                
            else:
                print('\nRegresa al menu y intente otra opcion')

        elif out == 5:
            print('Gracias por usar el programa Projecto 1!')
            break
        
        else:
            print('Entrada incorrecta, vuelve a seleccionar')

main()