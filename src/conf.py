import os




root = os.pardir
data = os.path.join(root, 'data')

paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'), 
    videos=os.path.join(data, 'videos'), iframes=os.path.join(data, 'iframes'),
    patches=os.path.join(data, 'patches'),
    traindata=os.path.join(data, 'iframes', 'traindata'),
    testdata=os.path.join(data, 'iframes', 'testdata'),

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
    pass



if __name__ == '__main__':
    main()