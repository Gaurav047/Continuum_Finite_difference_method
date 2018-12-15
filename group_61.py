%matplotlib inline
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour

L = 15
H = 10
h1 = 2
h2 = 4
N1 = 301
N2 = 201
print ("Enter the x1 component of velocity: ")
Vin = int(input())

vector_b = []
   
delx1 = float(L)/float(N1)
delx2 = float(H)/float(N2)
alpha = float(delx1)/float(delx2)
Vout = float(Vin*H)/float(h2-h1)
       
row = np.zeros(5*(N1+1)*(N2+1), dtype = int)
col = np.zeros(5*(N1+1)*(N2+1), dtype= int)
A = np.zeros(5*(N1+1)*(N2+1), dtype = float)
b = np.zeros((N1+1)*(N2+1), dtype = float)
    
row_num = 0
itr = 0

for i in range(1, N1):
    for j in range(1, N2):
      
        col[itr] = (i-1)*(N2+1)+j 
        row[itr] = row_num
        A[itr] = 1
        itr = itr + 1
        
        col[itr] = (i)*(N2+1)+j-1 
        row[itr] = row_num
        A[itr] = alpha*alpha
        itr = itr + 1
        
        col[itr] = (i)*(N2+1)+j 
        row[itr] = row_num
        A[itr] = -2*(1+alpha*alpha)
        itr = itr + 1
        
        col[itr] = (i)*(N2+1)+j+1 
        row[itr] = row_num
        A[itr] = alpha*alpha
        itr = itr + 1
        
        col[itr] = (i+1)*(N2+1)+j
        row[itr] = row_num
        A[itr] = 1
        itr = itr + 1
        
        vector_b.append(0)
        b[row_num] = 0            
        row_num = row_num + 1

for j in range(0, N2+1):

    col[itr] = j
    row[itr] = row_num
    A[itr] = 1
    itr = itr + 1
    
    vector_b.append(0)
    b[row_num] = 0
    row_num = row_num + 1
    


for j in range(N2-int(h2/delx2), N2 - int(h1/delx2)+1):

    col[itr] = N1*(N2+1)+j
    row[itr] = row_num
    A[itr] = 1
    itr = itr + 1
    
    vector_b.append(Vout*L)
    b[row_num] = Vout*L
    row_num = row_num + 1


for i in range(1, N1):

    col[itr] = i*(N2+1)+N2
    row[itr] = row_num
    A[itr] = 1
    itr = itr + 1
    
    col[itr] = i*(N2+1)+N2-1
    row[itr] = row_num
    A[itr] = -1
    itr = itr + 1
    
    vector_b.append(0)
    b[row_num] = 0
    row_num = row_num + 1
 

for i in range(1, N1):
   
    col[itr] = i*(N2+1)
    row[itr] = row_num
    A[itr] = 1
    itr = itr + 1
    
    col[itr] = i*(N2+1)+1
    row[itr] = row_num
    A[itr] = -1
    itr = itr + 1
    
    vector_b.append(0)
    b[row_num] = 0
    row_num = row_num + 1
    


for j in range(N2-int(h1/delx2)+1, N2+1):
   
    col[itr] = N1*(N2+1)+j
    row[itr] = row_num
    A[itr] = 1
    itr= itr + 1
    
    col[itr] = (N1-1)*(N2+1)+j
    row[itr] = row_num
    A[itr] = -1
    itr = itr + 1
    
    vector_b.append(0)
    b[row_num] = 0
    row_num = row_num + 1
    


for j in range(0, N2 - int(h2/delx2)):
   
    col[itr] = N1*(N2+1)+j
    row[itr] = row_num
    A[itr] = 1
    itr = itr + 1
    
    col[itr] = (N1-1)*(N2+1)+j
    row[itr] = row_num
    A[itr] = -1
    itr = itr + 1
    
    vector_b.append(0)
    b[row_num] = 0
    row_num = row_num + 1


matrix = csr_matrix((A, (row, col)), shape = ((N1+1)*(N2+1), (N1+1)*(N2+1)))

row = None
col = None
A = None


x = spsolve(matrix, vector_b)
k = 0

contour(np.transpose(np.reshape(x, (-1, N2+1))))
ty = np.transpose(np.reshape(x, (-1, N2+1)))
#print(ty)
ty.tofile('C:/Users/Gaurav kumar/Desktop/Data_Visualisation_With_Python/field.txt',sep='\r',format="%f")

vx=np.zeros((N1+1)*(N2+1),dtype=float)
vy=np.zeros((N1+1)*(N2+1),dtype=float)

for i in range(1,N1+1):
    for j in range(1,N2+1):
        index = i*(N2+1)+j
        vx[index] = (x[index]-x[index-N2-1])/delx1
        vy[index] = (x[index]-x[index-1])/delx2
        
        
for j in range(0,N2+1):
    index = j
    vy[index] = 0
    vx[index] = Vin
    
for i in range(1,N1+1):
    index = i*(N2+1)
    vy[index]=0
    vx[index]=(x[index]-x[index-N2-1])/delx1
    
#print(vy)
vy.tofile('C:/Users/Gaurav kumar/Desktop/Data_Visualisation_With_Python/vy.txt',sep='\r',format="%f")
#print(vx)
vx.tofile('C:/Users/Gaurav kumar/Desktop/Data_Visualisation_With_Python/vx.txt',sep='\r',format="%f")

xa,ya = np.linspace(0.0,L,N1+1),np.linspace(0.0,H,N2+1)
Xa,Ya = np.meshgrid(xa,ya)
Xa, Ya = Xa[0::5], Ya[0::5]
vx = np.reshape(vx,(N1+1,N2+1)); vx = np.transpose(vx)[0::5,0::5];
vy = np.reshape(vy,(N1+1,N2+1)); vy = np.transpose(vy)[0::5,0::5];
plt.figure()
cp2 = plt.quiver(Xa,Ya,vx,vy)
plt.show()