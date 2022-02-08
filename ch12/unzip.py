import sys
import gzip
import shutil
import os


if (sys.version_info >(3,0)):
    writemode = 'wb'
else:
    writemode='w'


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

path = get_base_dir('data')

print(path)
print(os.listdir(path))


    
zipped_mnist = [f for f in os.listdir(path) if f.endswith('ubyte.gz')]

for z in zipped_mnist:
    with gzip.GzipFile(path+'/'+z, mode='rb') as decompressed, open(path+'/'+z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 