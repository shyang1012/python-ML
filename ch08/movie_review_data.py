import os
import sys
import tarfile
import time
import urllib.request


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def reporthook(count, block_size, total_size):
    global start_time
    
    try:
        start_time = None
    except NameError:
        start_time = None

    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size

    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()

if __name__ == '__main__':
    source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    target = get_base_dir('data')+'/aclImdb_v1.tar.gz'
    path = get_base_dir('data')

    if not os.path.isdir(path):
        os.makedirs(path)

    if not os.path.isfile(target):
        urllib.request.urlretrieve(source, target, reporthook)

    tar_gz_file = tarfile.open(target,'r:gz')
    tar_gz_file.extractall(path =path)
    tar_gz_file.close()
    