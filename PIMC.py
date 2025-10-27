import numpy as np
import matplotlib.pyplot as plt


T=0.67
V_0=1.67
P= 1
dt=10**(-17)
epsilon=10**(-2.5)
n=2*10**5
K=P*T**2

def V(x) :
    return V_0*(x**2-1)**2


def simulation(n=n,T=T,K=K,epsilon=epsilon,P=P,dt=dt):
    protons_images=np.zeros(P)
    pos_mean=np.copy(protons_images)
    Pos=[np.copy(protons_images)]
    pos_centroide_mean=1/P*np.sum(protons_images)
    Pos_centroide=[pos_centroide_mean]
    acceptance_number=0
    choice_number=0
    U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
    Potential_energy=[U_t]
    for i in range(n):
        print(i)
        move=(np.random.randint(2,size=P)*2-1)*epsilon
        accept_proba=np.random.rand(P)
        for j in range(P):
            U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            #On regarde le nouvel état en ayant changé de position une image du proton
            protons_images[j]=protons_images[j] + move[j]
            U_t_new=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            delta_U_positive=U_t_new>U_t
            if delta_U_positive and accept_proba[j]>(np.exp(1/T*(U_t-U_t_new))):
                # La configuration n'est pas accepté et on revient à l'état précédent, sinon on ne change rien
                print("not accepted")
                choice_number +=1
                protons_images[j]=protons_images[j]-move[j]
            else:
                print("accepted")
                choice_number +=delta_U_positive
                acceptance_number+=delta_U_positive
        pos_centroide_mean=(pos_centroide_mean*+1/P*np.sum(protons_images))
        Pos_centroide.append(1/P*np.sum(protons_images))
        pos_mean=(pos_mean+1/P*protons_images)
        Pos.append(np.copy(protons_images))
        Potential_energy.append(1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)]))
    return pos_centroide_mean/n,Pos_centroide,pos_mean/n,Pos, Potential_energy, acceptance_number/choice_number


def choice_pas(Epsilon_list, n=n,T=T,K=K,P=P,dt=dt):
    acceptance_rate = []
    for epsilon in Epsilon_list:
        acceptance_rate.append(simulation(n, T, K, epsilon, P, dt)[-1])
    plt.plot(Epsilon_list, acceptance_rate)
    plt.xscale('log')
    plt.show()

def choice_n(N_list,T=T,K=K,epsilon=epsilon,P=P,dt=dt):
    for n in N_list:
        plt.plot(simulation( n=n,T=T,K=K,P=P,epsilon=epsilon,dt=dt)[1])
        plt.legend(n)
    plt.show()

def choice_P(P_list, n=n,T=T,K=K,epsilon=epsilon,dt=dt):
    for P in P_list:
        plt.plot(simulation( n=n,T=T,K=K,P=P,epsilon=epsilon,dt=dt)[-2])
        plt.legend(P)
    plt.show()
# pos_centroide_mean,Pos_centroide,pos_mean,Pos, Potential_energy=simulation()

# print(len([Pos[i][0] for i in range(n)]))
# print(pos_mean)
# plt.plot(Pos_centroide)
# print(np.dim(np.array([[Pos[i][j] for i in range(n)]for j in range(P)])))
# for j in range(P):
#     plt.plot([Pos[i][j] for i in range(n)])
# plt.plot(Pos)
# plt.plot(Potential_energy)
# plt.show()

choice_pas([10,5,2,1,0.5,0.3,10**(-1),10**(-2)])
choice_P(range(10))

choice_n([10**3, 10**4, 10*5, 10**6, 10**7])