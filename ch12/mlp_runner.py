import os
import struct
import numpy as np
import matplotlib.pyplot as plt

from ch12.neuralnet import NeuralNetMLP





def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
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
    
    n_epochs = 200
    
    nn = NeuralNetMLP(n_hidden=100, 
                  l2=0.01, 
                  epochs=n_epochs, 
                  eta=0.0005,
                  minibatch_size=100, 
                  shuffle=True,
                  seed=1)
    
    nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])
    print("\n")
    
    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.savefig(get_base_dir('images')+'/NeuralNetMLP.png', dpi = 300)
    # plt.show()
    plt.close()
    
    plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='Training')
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
            label='Validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.savefig(get_base_dir('images')+'/NeuralNetMLP_Accuracy.png', dpi = 300)
    # plt.show()
    plt.close()
    
    
    y_test_pred= nn.predict(X_test)
    
    acc = (np.sum(y_test == y_test_pred)
       .astype(np.float64) / X_test.shape[0])
    
    
    print('테스트 정확도: %.2f%%' % (acc * 100))
    
    
    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/NeuralNetMLP_miscl_img.png', dpi = 300)
    # plt.show()
    plt.close()
    