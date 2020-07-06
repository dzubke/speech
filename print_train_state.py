import pickle
import sys

def main(train_state_path:str):
    with open(train_state_path, 'rb') as fid:
        print(pickle.load(fid))

if __name__ == "__main__":
    train_state_path = sys.argv[1]
    main(train_state_path)
