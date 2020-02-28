import pickle
if __name__=='__main__':
    len = 0
    with open('results.json', 'rb') as f:
        a = pickle.load(f)
    print(a)