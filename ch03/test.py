import os


if __name__ == '__main__':
    path1 = os.path.dirname(os.path.abspath(__file__))

    print( os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
    