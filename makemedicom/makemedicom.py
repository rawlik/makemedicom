import h5py
import pydicom


def hello():
    print("hello!")
    print(h5py.__version__)
    print(pydicom.__version__)
    print("hello")


if __name__ == "__main__":
    hello()
