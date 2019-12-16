## IA TD5

import math
import csv
import numpy as np

## Chargement du fihier .data

with  open('C:/Users/corentin/Desktop/TP5IA/iris.data','r') as csvfile: # Attention changer le chemin d'accès si besoin 
    lines = csv.reader(csvfile)
    dataset=list(lines)

## Construction de la fonction distance Eulérienne

#Retourne le distance eulérienne de 2 données présent dans le fichier iris.data
# Le programme séléctionne ces 2 données grace à leur indice respectif.

def getDistance(indice1,indice2):
    distance =0
    for x in range(3): # de 0 à 3 car on a 4 dimensions
        distance += pow((float(dataset[indice1][x])-float(dataset[indice2][x])),2)
    return math.sqrt(distance)


## Construction de la fonction de Déduction      

#Ce programme compart à ca base de donnée c'est à dire 25 fleurs de chaque type une fleur de type inconnue et donne sa deduction concernant sa nature.
# On a choisit les 25 premières donnée de chaque fleur comme base d'apprentissage pour le programme les 25 autres de chaque type sont utilisées pour le test.

def deduction(item,k):
    compteursetosa=0
    compteurversicolor=0
    compteurvirginica=0
    for i in range(24):
        if(getDistance(item,i)<k):
            compteursetosa+=1
    for i in range(51,75):
        if(getDistance(item,i)<k):
            compteurversicolor+=1
    for i in range(101,125):
        if(getDistance(item,i)<k):
            compteurvirginica+=1
    if((compteursetosa>compteurversicolor)&(compteursetosa>compteurvirginica)):
        return "Iris-setosa"
    if((compteurversicolor>compteursetosa)&(compteurversicolor>compteurvirginica)):
        return "Iris-versicolor"
    if((compteurvirginica>compteurversicolor)&(compteurvirginica>compteursetosa)):
        return "Iris-virginica"
        
    return deduction(item,k+0.01) # En cas d'égalité on elargit un petit peu le cercle ( paramètre k )
    

## Construction de la fonction permettant d'obtenir la Matrice de Convolution


#Cette fonction prend les 25 données de chaque type de fleur restante et compare la déduction de l'IA vis à vis de la réalité.
# Les résultats sont affichés sous forme d'une matrice de convolution
# Première colonne: Iris Setosa Réalité
#Deuxième colonne: Iris Versicolor Réalité
#Troisième colone: Iris Virginica Réalité
#Première ligne: Iris Setosa Déduction
#Deuxième ligne: Iris Versicolor Déduction
#Troisième ligne: Iris Virginica Déduction

def MatriceConvolution(k):
    Mat=np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(25,50):
        res=deduction(i,k)
        restrue=dataset[i][4]
        if(res==restrue):
            
            if(res=="Iris-setosa"):
                Mat[0][0]+=1
                
            if(res=="Iris-versicolor"):
                Mat[1][1]+=1
                
            if(res=="Iris-virginica"):
                Mat[2][2]+=1
                
        else:
            if(restrue=="Iris-setosa"):
                if(res=="Iris-versicolor"):
                   Mat[1][0]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][0]+=1
                    
            if(restrue=="Iris-versicolor"):
                if(res=="Iris-setosa"):
                   Mat[0][1]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][1]+=1
                    
            if(restrue=="Iris-virginica"):
                if(res=="Iris-versicolor"):
                   Mat[1][2]+=1 
                if(res=="Iris-setosa"):
                    Mat[0][2]+=1
                    
    for i in range(75,100):
        res=deduction(i,k)
        restrue=dataset[i][4]
        if(res==restrue):
            
            if(res=="Iris-setosa"):
                Mat[0][0]+=1
                
            if(res=="Iris-versicolor"):
                Mat[1][1]+=1
                
            if(res=="Iris-virginica"):
                Mat[2][2]+=1
                
        else:
            if(restrue=="Iris-setosa"):
                if(res=="Iris-versicolor"):
                   Mat[1][0]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][0]+=1
                    
            if(restrue=="Iris-versicolor"):
                if(res=="Iris-setosa"):
                   Mat[0][1]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][1]+=1
                    
            if(restrue=="Iris-virginica"):
                if(res=="Iris-versicolor"):
                   Mat[1][2]+=1 
                if(res=="Iris-setosa"):
                    Mat[0][2]+=1
                    
    for i in range(125,150):
        res=deduction(i,k)
        restrue=dataset[i][4]
        if(res==restrue):
            
            if(res=="Iris-setosa"):
                Mat[0][0]+=1
                
            if(res=="Iris-versicolor"):
                Mat[1][1]+=1
                
            if(res=="Iris-virginica"):
                Mat[2][2]+=1
                
        else:
            if(restrue=="Iris-setosa"):
                if(res=="Iris-versicolor"):
                   Mat[1][0]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][0]+=1
                    
            if(restrue=="Iris-versicolor"):
                if(res=="Iris-setosa"):
                   Mat[0][1]+=1 
                if(res=="Iris-virginica"):
                    Mat[2][1]+=1
                    
            if(restrue=="Iris-virginica"):
                if(res=="Iris-versicolor"):
                   Mat[1][2]+=1 
                if(res=="Iris-setosa"):
                    Mat[0][2]+=1
                    
    return Mat


## Lancement de la fonction Matrice de Convolution
    
print(MatriceConvolution(0.5))
    
    
    
    
    
    
    
    
    
    