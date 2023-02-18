import os
from os.path import expanduser



root = os.pardir
data = os.path.join(root, 'data')

paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'), 
    videos=os.path.join(data, 'videos'), iframes=os.path.join(data, 'iframes'),
    train=os.path.join(data, 'dataset', 'train'), 
    val=os.path.join(data, 'dataset', 'val'), test=os.path.join(data, 'dataset', 'test')

)


def ds_rm(array: list):
    try:
        array.remove('DS_Store')
    except Exception as e:
        print(e)

    return array

def creatdir(path):
    try:
        os.makedirs(path)
    except Exception as ex:
        print(ex)



def main():
    for k, v in paths.items():
        creatdir(v)


if __name__ == '__main__':
    main()