import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob
from util import scale2uint8

class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(x_fn, y_fn, mb_size, in_depth, img_size):
    X, Y = None, None
    with h5py.File(x_fn, 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)

    with h5py.File(y_fn, 'r') as hdf_fd:
        Y = hdf_fd['images'][:].astype(np.float32)

    while True:
        idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
        rst = np.random.randint(0, X.shape[1]-img_size, mb_size)
        cst = np.random.randint(0, X.shape[2]-img_size, mb_size)

        batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
        batch_X = [batch_X[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]

        batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3)
        batch_Y = [batch_Y[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]

        yield np.array(batch_X), np.array(batch_Y)

def get1batch4test(x_fn, y_fn, in_depth):
    X = h5py.File(x_fn, 'r')['images']
    Y = h5py.File(y_fn, 'r')['images']

    idx = (X.shape[0]//2, )
    batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
    batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3) 

    return batch_X.astype(np.float32) , batch_Y.astype(np.float32)


