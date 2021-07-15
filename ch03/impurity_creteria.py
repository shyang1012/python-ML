import matplotlib.pyplot as plt
import numpy as np
import os

def gini(p):
    return (p)*(1-(p)) +(1-p)*(1-(1-p))


def entropy(p):
    return -p*np.log2(p) - (1- p)*np.log2((1-p))

def errors(p):
    return 1 - np.max([p, 1-p])

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

if __name__ == '__main__':
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p !=0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [errors(i) for i in x]

    fig = plt.figure()
    ax = plt.subplot(111)

    for i, lab, ls, c in zip([ent, sc_ent, gini(x), err]
        ,['Entropy', 'Entropy(scaled)', 'Gini impurity', 'Misclassification error']
        ,['-','-','--','-']
        ,['black','lightgray','red','green','cyan']
    ):
        line = ax.plot(x, i, label= lab, linestyle=ls, lw= 2, color=c)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('impurity index')
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(get_base_dir('images')+'/impurity_creteria.png', dpi = 300)
    plt.show()
    plt.close()
