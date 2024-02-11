import math
import numpy as np 
import matplotlib.pyplot as plt

def normal_distribution(x, mean, sd):
    return (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)

def example1():
    x = np.linspace(1, 100, 1000)


    mean = np.mean(x)
    sd = np.std(x)

    pdf = normal_distribution(x, mean, sd)

    print("X:\n", x)
    print("mean:\n", mean)
    print("sd:\n", sd)
    print("pdf:\n", pdf)

    # _, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.plot(x, x, color='red')
    # ax1.set_title("linespace range")

    # ax2.plot(x, pdf, color='blue')
    # ax2.set_title("pdf")  

    # plt.show()

    plt.plot(x, x, color='red')
    plt.plot(x, pdf, color='blue')
    plt.show()

def kl_divergence(p,q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))































































































































def kl_example(): 
    x = np.linspace(1, 100, 1000)
    y = np.linspace(50, 150, 1000)
    
    mu1 = np.mean(x)
    sigma1 = np.std(x)

    mu2 = np.mean(y)
    sigma2 = np.std(y)

    p = normal_distribution(x, mu1, sigma1)
    q = normal_distribution(y, mu2, sigma2)

    kl_divergence_value = kl_divergence(p, q)

    print("KL Divergence_value: ", kl_divergence_value)

    plt.plot(x, x, color='red')
    plt.plot(y, y, color='blue')
    plt.plot(x, p, color='green')
    plt.plot(y, q, color='yellow')
    plt.show()

kl_example()