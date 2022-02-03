from pylab import *
import perceptron_data as data 



# 1 # implimentation de l’algorithme du perceptron binaire

def perceptron_Binaire(List_Donnees_S,n):
    nombre_Attributs=len(List_Donnees_S[0][0])  # calculer le nombre d’attributs des données d’entrée
    w=[0,]*nombre_Attributs # Initialisation de vecteur de poid
    for i in range(n):
        for (x,y) in List_Donnees_S:
            y_ch=signe(w,x)
            if (y_ch != y): # ajouster la valeur w
                if(y== False):
                    for (index,xj,wj) in zip(range(len(w)),x,w) :
                        w[index]=wj-xj
                else:
                    for (index,xj,wj) in zip(range(len(w)),x,w) :
                        w[index]=wj+xj
    return w


def produit_Scalaire(w,x): # calculer le produit scalaire entre les deux vecteurs w et x
    p=0
    for (xj,wj) in zip(x,w) :
        p= p+xj*wj
    return p
                
def signe(w,x): #calculer le signe de produit scalaire
    
    p1=produit_Scalaire(w,x)
    if(p1<0):
        return False
    else:
        return True





# 2 # Géneration d'un jeux de données et application de l'algorithme de perceptron binaire
#fonction qui génére un jeu de données 2D linéairement séparable de taille n.
def generer_Donnees(n):
    x1b = (rand(n)*2-1)/2-0.5
    x2b = (rand(n)*2-1)/2+0.5
    x1r = (rand(n)*2-1)/2+0.5
    x2r = (rand(n)*2-1)/2-0.5
    donnees = []
    for i in range(len(x1b)):
        donnees.append(((x1b[i],x2b[i]),True))
        donnees.append(((x1r[i],x2r[i]),False))
    return donnees

#Generation d'un jeu  de données
nombre_Donnees=int(input("donnez Le nombre de donnes: "))
n=int(input("donnez n : "))
donnees_G=generer_Donnees(nombre_Donnees)  # tout jeux donné 
indice=int(nombre_Donnees*0.7)
donnees_Apprentissage= donnees_G[0:indice]  #jeux données d'aprentissage 
donnees_Test= donnees_G[indice+1:nombre_Donnees-1]   #jeux données pour le test

#Application de l'algorithme de perceptron binaire sur les donnees d'apprentissage
Nouveaux_W=perceptron_Binaire(donnees_Apprentissage,n)




# defition de la fonction permatant de calculer l'erreur de prédiction de classifieur
def erreur_Prediction(w,List_Donnees_S):
    erreur=0
    for (x,y) in List_Donnees_S:
        y_ch=signe(w,x)
        if (y_ch != y):
            erreur=erreur+1
    return erreur/len(List_Donnees_S)*100


#calculer l'erreur de prediction de calassifieur
print("l'erreur prédiction de classifieur sur les donneés d'apprentissage = ",erreur_Prediction(Nouveaux_W,donnees_Apprentissage),"%")
print("l'erreur prédiction de classifieur sur les donneés de test = ",erreur_Prediction(Nouveaux_W,donnees_Test),"%")



#3 # Représentation graphique 

print("Le nouveaux W est:",Nouveaux_W)

#representation graphique des donneés de test
for ((x,y),label) in donnees_Test:

    if(label==True): 
        plot(x, y, "1", label="line 1", color="r")
    else:
        plot(x, y, "2", label="line 2", color="b")

plt.title("representation graphique des donneés de test")
a= array([-1, 1])
b= array([0, 0])
plot(a, b)

    
    

#representation graphique de classifieur ????????????????????????????????????????????????????????


axis([-1, 1, -1, 1])
show()



#4 # perceptron binaire avec le biais 


# modifier l’algorithme du perceptron binaire afin de tenir compte lors de l’apprentissage le biais du classifieur linéaire.
def perceptron_Binaire_Bias(List_Donnees_S,n,b):
    w=0 # Initialisation de vecteur de poid
    for i in range(n):
        for (x,y) in List_Donnees_S:
            y_ch=signe_Bias(x*w+b)
            if (y_ch != y): # ajouster la valeur w
                if(y== False):
                    w=w-x
                else:
                    w=w+x
    return w

def signe_Bias(p1): #calculer le signe de produit scalaire
    if(p1<0):
        return False
    else:
        return True


# Appliquer l’algorithme modifié sur le jeu de données data.biais 
Nouveaux_W1=perceptron_Binaire_Bias(data.bias,n,-1)
print("Aprés les modification sur le jeu de données data.biais le nouveaux W est:",Nouveaux_W1)

# montrer qu’il est capable de le séparer linéairement.????????????????????????????????????????
indice_b=int(len(data.bias)*0.1)
donnees_Test_b= data.bias[0:indice_b] 



for (x,y) in donnees_Test_b:

    if(y==True): 
        plot(x, "1", label="line 1", color="r")
    else:
        plot(x, "2", label="line 2", color="b")

plt.title("representation graphique des donneés de t")
a= array([-0.1, 0.1])
b= array([1, 1])

plot(a, b)
axis([-0.1, 0.1, 0, 2])
show()

