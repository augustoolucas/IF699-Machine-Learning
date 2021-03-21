#Bibliotecas utilizadas
import misc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#modulos implementados
from LVQ import lvq1
from LVQ import lvq2_1
from LVQ import lvq3

#-------------Utilizando um dataset artificial com apenas dois features para ilustrar o funcionamento do LVQ.-------------------
dataset = misc.loadData('long1.arff')
dataset = np.asarray(dataset)

plt.scatter(dataset[:,0],dataset[:,1], c = dataset[:,2])
plt.title("2d Dataset", fontsize='large')
plt.show()

#---------------------------------------Escolha de um prototipo aleatorio de cada classe----------------------------------------
prototypes = misc.getRandomPrototypes(dataset,1)

plt.scatter(dataset[:,0],dataset[:,1], c = dataset[:,2])
plt.title("Prototipos: azul e verde", fontsize='large')
plt.scatter(prototypes[:,0],prototypes[:,1], c = ['g','r'])
plt.show()

#-----------------------------Ajuste dos prototipos utilizado o lvq1 com 0.4 de taxa de aprendizado-----------------------------
prototypes = lvq1(prototypes,dataset,0.4)

plt.title("Prototipos: azul e verde(ajustados)", fontsize='large')
plt.scatter(dataset[:,0],dataset[:,1], c = dataset[:,2])
plt.scatter(prototypes[:,0],prototypes[:,1], c = ['g','r'])
plt.show()

print "------------------------------Verificacao LVQ1 com 5 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ1 com dataset CM1 do repositorio promise--------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #-----------------------------Ajuste dos prototipos utilizado o lvq1 com alfa inicial = 0.4---------------------------------
    prototypes = lvq1(prototypes,trainset,0.4)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ1 com dataset KC1 do repositorio promise--------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #-----------------------------Ajuste dos prototipos utilizado o lvq1 com alfa inicial = 0.4---------------------------------
    prototypes = lvq1(prototypes,trainset,0.3)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ 3-nn:" + repr(media2/10.0)

print "------------------------------Verificacao LVQ1 com 10 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ1 com dataset CM1 do repositorio promise--------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 5 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #-----------------------------Ajuste dos prototipos utilizado o lvq1 com alfa inicial = 0.4---------------------------------
    prototypes = lvq1(prototypes,trainset,0.4)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ1 com dataset KC1 do repositorio promise--------------------------------------------
dataset = misc.loadData('KC1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 5 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #-----------------------------Ajuste dos prototipos utilizado o lvq1 com alfa inicial = 0.4---------------------------------
    prototypes = lvq1(prototypes,trainset,1)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ 3-nn:" + repr(media2/10.0)

print "------------------------------Verificacao LVQ2.1 com 5 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ2.1 com dataset CM1 do repositorio promise------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #-------------------------Ajuste dos prototipos utilizado o lvq2.1 com alfa inicial = 0.4 e w=0.6-----------------------------
    prototypes = lvq2_1(prototypes,trainset,0.4,0.6)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ2.1 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ2.1 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ2.1 com dataset KC1 do repositorio promise--------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #-----------------------Ajuste dos prototipos utilizado o lvq2.1 com alfa inicial = 0.4 e w=0.6-----------------------------
    prototypes = lvq2_1(prototypes,trainset,0.4,0.6)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ2.1 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ2.1 3-nn:" + repr(media2/10.0)

print "------------------------------Verificacao LVQ2.1 com 10 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ2.1 com dataset CM1 do repositorio promise-----------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino-----------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 5 prototipos aleatório de cada classe-----------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #-------------------------Ajuste dos prototipos utilizado o lvq2.1 com alfa inicial = 0.4 e w=0.6-----------------------------
    prototypes = lvq2_1(prototypes,trainset,0.4,0.6)

    #-------------------------------------------Separação instancia/classe-----------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn-----------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn-----------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ2.1 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ2.1 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ2.1 com dataset KC1 do repositorio promise------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino-----------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 5 prototipos aleatório de cada classe-----------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #-------------------------Ajuste dos prototipos utilizado o lvq1 com alfa inicial = 0.4 e w=0.6-----------------------------
    prototypes = lvq2_1(prototypes,trainset,0.4,0.6)

    #-------------------------------------------Separação instancia/classe-----------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn-----------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn-----------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ2.1 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ2.1 3-nn:" + repr(media2/10.0)

print "------------------------------Verificacao LVQ3 com 5 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ3 com dataset CM1 do repositorio promise--------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #---------------------Ajuste dos prototipos utilizado o lvq3 com alfa inicial = 0.4, w=0.6 e e=0.5--------------------------
    prototypes = lvq3(prototypes,trainset,0.4,0.6,0.5)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ3 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ3 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ2.1 com dataset KC1 do repositorio promise--------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,5)

    #---------------------Ajuste dos prototipos utilizado o lvq3 com alfa inicial = 0.4, w=0.6 e e=0.5--------------------------
    prototypes = lvq3(prototypes,trainset,0.4,0.6,0.5)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ3 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ3 3-nn:" + repr(media2/10.0)

print "------------------------------Verificacao LVQ3 com 10 prototipos por classe-----------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------LVQ3 com dataset CM1 do repositorio promise--------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #---------------------Ajuste dos prototipos utilizado o lvq3 com alfa inicial = 0.4, w=0.6 e e=0.5--------------------------
    prototypes = lvq3(prototypes,trainset,0.4,0.6,0.5)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ3 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ3 3-nn:" + repr(media2/10.0)

#----------------------------------------LVQ2.1 com dataset KC1 do repositorio promise--------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #---------------------------------------Escolha de 3 prototipos aleatório de cada classe------------------------------------
    prototypes = misc.getRandomPrototypes(trainset,10)

    #---------------------Ajuste dos prototipos utilizado o lvq3 com alfa inicial = 0.4, w=0.6 e e=0.5--------------------------
    prototypes = lvq3(prototypes,trainset,0.4,0.6,0.5)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(prototypes_instances , prototypes_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media LVQ3 1-nn:" + repr(media1/10.0)
print "Acuracia media LVQ3 3-nn:" + repr(media2/10.0)

print "-----------------------------------------------Verificacao kNN---------------------------------------------"
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------kNN com dataset CM1 do repositorio promise--------------------------------------------
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    trainset_instances = trainset[:,range(len(trainset[0])-1)]
    trainset_class = trainset[:,-1]
    
    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(trainset_instances , trainset_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(trainset_instances , trainset_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media 1-nn:" + repr(media1/10.0)
print "Acuracia media 3-nn:" + repr(media2/10.0)

#----------------------------------------kNN com dataset KC1 do repositorio promise--------------------------------------------

dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

media1 = 0
media2 = 0
for aux in range(10):
    #-------------------------------Particionando o dataset aleatoriamente em 30% teste 70% treino------------------------------
    trainset, testset = train_test_split(dataset, test_size=0.30)

    #-------------------------------------------Separação instancia/classe------------------------------------------------------
    trainset_instances = trainset[:,range(len(trainset[0])-1)]
    trainset_class = trainset[:,-1]
    
    testset_instances = testset[:,range(len(testset[0])-1)]
    testset_class = testset[:,-1]

    #-------------------------------------------Verificação de acuracia com 1-nn------------------------------------------------
    #treinamento
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(trainset_instances , trainset_class)
    #score
    media1 += knn.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ3 1-nn: "+repr(knn.score(testset_instances,testset_class) * 100) + '%'

    #-------------------------------------------Verificação de acuracia com 3-nn------------------------------------------------

    #treinamento
    knn2 = KNeighborsClassifier(n_neighbors=3)  
    knn2.fit(trainset_instances , trainset_class)
    #score
    media2 += knn2.score(testset_instances,testset_class) * 100
    #print "Acuracia LVQ2.1 3-nn: "+repr(knn2.score(testset_instances,testset_class) * 100) + '%'

print "Acuracia media 1-nn:" + repr(media1/10.0)
print "Acuracia media 3-nn:" + repr(media2/10.0)