import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def load_mnist(path, kind='train'):
    """`path`에서 MNIST 데이터 불러오기"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels


if __name__ == '__main__':
    X_train, y_train = load_mnist(get_base_dir('data'), kind='train')
    print('행: %d, 열: %d' % (X_train.shape[0], X_train.shape[1]))
   
    X_test, y_test = load_mnist(get_base_dir('data'), kind='t10k')
    print('행: %d, 열: %d' % (X_test.shape[0], X_test.shape[1]))
    
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
        
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/mnist-sample.png', dpi = 300)
    # plt.show()
    plt.close()
    
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/mnist-sample-7.png', dpi = 300)
    # plt.show()
    plt.close()
    
    np.savez_compressed(get_base_dir('data')+'/mnist_scaled.npz', X_train= X_train, y_train= y_train, X_test=X_test, y_test=y_test)
    
    mnist = np.load(get_base_dir('data')+'/mnist_scaled.npz')
    print(mnist.files)
    
    X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]
    
    del mnist
    
    print(X_train.shape)