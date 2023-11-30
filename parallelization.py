
import numpy as np
from time import time
from threading import Thread
from time import time, sleep
from queue import Queue

def func(x):
    sleep(0.0005)
    return x * x

class Worker(Thread):
    def __init__(self, queue, result):
        Thread.__init__(self)
        self.queue = queue
        self.result = result

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            i, elmt = self.queue.get()
            try:
                self.result[i] = func(elmt)
            finally:
                self.queue.task_done()

my_array = list(2 * np.ones(1000))

def parallel(my_array):
    
    result = list(np.zeros(1000))
    queue = Queue()

    # Create 8 worker threads
    for x in range(8):
        worker = Worker(queue, result)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
    # Put the tasks into the queue as a tuple
    for i,elmt in enumerate(my_array):
        queue.put((i, elmt))
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
st1 = time()
for i in range(len(my_array)):
    my_array[i] = func(my_array[i])
end1 = time()
print("Iterative: " + str(end1-st1))

st2 = time()
results = parallel(my_array.copy())
end2 = time()

print("Parallel: "+str(end2-st2))
# print("Original Array:", my_array)
# print("Results Array:", results)