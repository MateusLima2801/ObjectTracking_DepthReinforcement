from threading import Thread
from queue import Queue 
from time import time, sleep
import numpy as np

class JobWorkers():
    def __init__(self, queue: Queue, func, num_workers=1, no_print = False, *args):
        self.num_workers = num_workers
        self.queue = queue
        self.func = func
        self.args = args
        self.print = not no_print
        start = time()
        self.start_workers()
        self.queue.join()
        end = time()
        print(f"All done in {end-start}s.")

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            item = self.queue.get()
            if self.print:
                print(f'\nWorking item: {item}')
            self.func(item, self.args) 
            self.queue.task_done()


    
# s = "Hello World"
# w = ", once again"

# def blokkah(el, args):
#     sleep(5)
#     print(f"{''.join(args)} {el}")
          
# q = Queue()
# for i in range(20):
#     q.put(i)


# j = JobWorkers(q, blokkah, 5, s, w)


