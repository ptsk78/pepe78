import random
import matplotlib.pyplot as plt
import math
from multiprocessing import cpu_count, Array, Process

def get_average(a):
    return sum(a)/len(a)

def get_std_average(a):
    av = get_average(a)
    ret = 0
    for i in range(len(a)):
        tmp = a[i] - av
        ret += tmp * tmp
    
    return math.sqrt(ret / len(a)), av

def get_random_transf(transf):
    return transf(random.random())

def transf1(x):
    return x
    
def transf2(x):
    return x ** 3

def ThreadFunc(N1, N2, transf, a, threadNum, numThreads):
    for j in range(threadNum, N1, numThreads):
        tmp = []
        for i in range(N2):
            tmp.append(get_random_transf(transf))
        a[j] = get_average(tmp)

for transf in [transf1, transf2]:
    plt.figure(num='Central Limit Theorem', figsize = (20,10))
    N1 = 100000
    N2 = 10000

    a = []
    for i in range(N1):
        a.append(get_random_transf(transf))

    std, ave = get_std_average(a)
    std2 = std / math.sqrt(N2)

    plt.subplot(1,2,1)
    plt.title('Input probability density function (pdf)')
    plt.hist(a, bins=75, density=True, label='measured')
    plt.legend()

    # no need for lock as every thread re-writes different one
    a = Array('d', N1, lock=False)
    threads = []
    print('Number of threads:', cpu_count())
    for i in range(cpu_count()):
        # threads in Python don't behave same way as in C#, hence using Process
        thread = Process(target=ThreadFunc, args=(N1, N2, transf, a, i, cpu_count(),))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    mina = min(a)
    maxa = max(a)
    ave2 = get_average(a)

    x = []
    y = []
    for i in range(1001):
        t = mina + i * (maxa-mina) / 1000.0
        x.append(t)
        y.append(math.exp(-0.5 * (((t-ave2)/std2) ** 2)) / (std2 * math.sqrt(2 * math.pi)))

    plt.subplot(1,2,2)
    plt.title("Input's sequences average probability density function (pdf)")
    plt.hist(a, bins=75, density=True, label='measured')
    plt.rcParams['text.usetex'] = True
    plt.plot(x, y, label=r'theoretical $\sigma_{new} = \frac{\sigma}{\sqrt{N_2}}$')
    plt.legend()
    plt.show()
