import numpy as np
import matplotlib.pyplot as plt


T=(1/3)
V_0=(5/3)
P= 1
dt=10**(-17)
epsilon=0.01
n=2*10**4
K=P*T**2

def V(x) :
    return V_0*(x**2-1)**2


def simulation(n=n,T=T,K=K,epsilon=epsilon,P=P,dt=dt):
    protons_images=np.zeros(P)
    pos_mean=np.copy(protons_images)
    Pos=[np.copy(protons_images)]
    Pos_centroide=[1/P*np.sum(protons_images)]
    acceptance_number=0
    choice_number=0
    U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
    Potential_energy=[U_t]
    for i in range(n):
        # print(i)
        move=(np.random.randint(2,size = P)*2-1)*epsilon
        accept_proba=np.random.rand(P)
        for j in range(P):
            U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            #On regarde le nouvel état en ayant changé de position une image du proton
            protons_images[j]=protons_images[j] + move[j]
            U_t_new=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            delta_U_positive=U_t_new>U_t
            if delta_U_positive and accept_proba[j]>(np.exp(1/T*(U_t-U_t_new))):
                # La configuration n'est pas accepté et on revient à l'état précédent, sinon on ne change rien
                choice_number +=1
                protons_images[j]=protons_images[j]-move[j]
            else:
                choice_number +=delta_U_positive
                acceptance_number+=delta_U_positive
        Pos_centroide.append(1/P*np.sum(protons_images))
        pos_mean=(pos_mean+protons_images)
        Pos.append(np.copy(protons_images))
        Potential_energy.append(1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)]))
    return np.sum(Pos_centroide)/n,Pos_centroide,pos_mean/n,Pos, Potential_energy, acceptance_number/choice_number


def choice_pas(Epsilon_list, n=n,T=T,K=K,P=P,dt=dt):
    acceptance_rate = []
    Avg_position = []
    for epsilon in Epsilon_list:
        results = simulation(n, T, K, epsilon,P, dt)
        plt.plot(results[-2], label=f"{str(epsilon)}")
    plt.legend()
    print(Avg_position)
    plt.show()

def choice_n(N_list,T=T,K=K,epsilon=epsilon,P=P,dt=dt):
    Avg_position = []
    for n in N_list:
        results= simulation( n=n,T=T,K=K,epsilon=epsilon,P=P,dt=dt)
        Avg_position.append(results[0])
        plt.plot(results[1], label=f"{str(n)}")
        plt.legend()
    print(Avg_position)
    plt.show()

def choice_P(P_list, n=n,T=T,K=K,epsilon=epsilon,dt=dt):
    Average_energy=[]
    for P in P_list:
        Potential_Energy = simulation( n=n,T=T,K=K,epsilon=epsilon,P=P, dt=dt)[-2]
        plt.plot(Potential_Energy, label=f"{str(P)}")
        plt.legend()
        Average_energy.append(np.sum(Potential_Energy)/n)
    plt.show()
    plt.plot(P_list, Average_energy)
    plt.show()
# pos_centroide_mean,Pos_centroide,pos_mean,Pos, Potential_energy, _ =simulation()

# print(len([Pos[i][0] for i in range(n)]))
# print(pos_centroide_mean)
# print(pos_mean)
# plt.plot(Pos_centroide)
# # print(np.dim(np.array([[Pos[i][j] for i in range(n)]for j in range(P)])))
# for j in range(P):
#     plt.plot([Pos[i][j] for i in range(n)])
# plt.plot(Pos)
# # plt.plot(Potential_energy)
# plt.show()

choice_pas([0.003,0.004,0.005])
# choice_P(range(1,10))

#choice_n([10**3, 10**4, 10*5, 10**6, 10**7])


def choose_displacement(n=n,T=T,K=K,epsilon=epsilon,P=P,dt=dt):
    currrent_epsilon = epsilon
    previous_right =False
    delta_epsilon = 0.1
    Epsilon=[currrent_epsilon]
    protons_images=np.zeros(P)+1
    pos_mean=np.copy(protons_images)
    Pos=[np.copy(protons_images)]
    Pos_centroide=[1/P*np.sum(protons_images)]
    acceptance_number=0
    choice_number=0
    U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
    for i in range(n):
        print(currrent_epsilon)
        move=(np.random.randint(2,size = P)*2-1)*currrent_epsilon
        accept_proba=np.random.rand(P)
        for j in range(P):
            U_t=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            #On regarde le nouvel état en ayant changé de position une image du proton
            protons_images[j]=protons_images[j] + move[j]
            U_t_new=1/P*np.sum(np.array(list(map(V,protons_images))))+1/2*K*np.sum([(protons_images[i]-protons_images[(i+1)%P])**2 for i in range(P)])
            delta_U_positive=U_t_new>U_t
            if delta_U_positive and accept_proba[j]>(np.exp(1/T*(U_t-U_t_new))):
                # La configuration n'est pas accepté et on revient à l'état précédent, sinon on ne change rien
                choice_number +=1
                protons_images[j]=protons_images[j]-move[j]
            else:
                choice_number +=delta_U_positive
                acceptance_number+=delta_U_positive
        if i!=0 and i%1000==0:
            if acceptance_number/choice_number >0.45:
                if not previous_right:
                    delta_epsilon = delta_epsilon/2
                currrent_epsilon = epsilon + delta_epsilon
                previous_right=True
            elif acceptance_number/choice_number <0.40:
                if previous_right:
                    delta_epsilon = delta_epsilon/2
                currrent_epsilon = epsilon - delta_epsilon
                previous_right = False
            else:
                currrent_epsilon = currrent_epsilon + (2*previous_right -1)*delta_epsilon
            print("change")
            acceptance_number =0
            choice_number=0
        Epsilon.append(currrent_epsilon)
        Pos_centroide.append(1/P*np.sum(protons_images))
        pos_mean=(pos_mean+protons_images)
        Pos.append(np.copy(protons_images))
    return currrent_epsilon, Epsilon, Pos_centroide, pos_mean, Pos

# currrent_epsilon, Epsilon, Pos_centroide, pos_mean, Pos = choose_displacement()
# print(currrent_epsilon)
# plt.plot(Pos)
# plt.show()
# plt.plot(Epsilon)
# plt.show()

