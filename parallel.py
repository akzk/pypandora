from multiprocessing import cpu_count, Process
from abc import ABCMeta, abstractmethod
from functools import reduce
import pickle
import types
import time
import os

def run(infile, mapper, reducer, max_cores=None):
    master = Master(infile, mapper, reducer, max_cores)
    return master.run()

class Mapper(metaclass=ABCMeta):
    """
        mapper
    """
    def create(self, x):
        """ will be used once when func combine() exists """
        return x
    @abstractmethod
    def map(self):
        pass
    def destory(self):
        pass

class Reducer(metaclass=ABCMeta):
    """
        reducer
    """
    def create(self, x):
        """ will be used once when getting the first map result """
        return x
    @abstractmethod
    def reduce(self):
        pass
    def destory(self):
        pass

class SimpleFileReducer(Reducer):
    """
        a reducer outputs result to a file directly and return None
    """
    def __init__(self, outfile):
        self.writer = open(outfile, 'w', encoding='utf-8')
        assert self.writer != None
    def create(self, a):
        self.writer.write(a)
    def reduce(self, a, b):
        self.writer.write(b)
    def destory(self):
        self.writer.close()

class SimpleMapper(Mapper):
    """
        simple mapper
    """
    def map(self):
        pass

class SimpleReducer(Reducer):
    """
        simple reducer
    """
    def reduce(self):
        pass

class ShuffleIO(object):
    """
        data io, local file
    """
    @staticmethod
    def write_to_file(data_fp, index_fp, x):
        """
            write x to file with pickle format
        """
        data = pickle.dumps(x)
        size = len(data)
        data_fp.write(data)
        index_fp.write("%d\n" % size)
    
    @staticmethod
    def read_from_file(data_fp, index_fp):
        """
            iterator, read data from file with pickle format
        """
        for line in index_fp:
            size = int(line.strip())
            data = pickle.loads(data_fp.read(size))
            yield data

class Worker(object):
    """
        worker, using 1 core
    """
    def __init__(self, infile, task_name, proc_num, start, length, mapper):
        self.infile = infile
        self.task_name = task_name
        self.proc_num = proc_num
        self.start = start
        self.length = length
        self.mapper = mapper
        self.data_file_name = self.filename("data")
        self.index_file_name = self.filename("index")

    def filename(self, target):
        """
            return the filename corresponding to target
        """
        if target == "data":
            return "%s_%d.data" % (self.task_name, self.proc_num)
        elif target == "index":
            return "%s_%d.index" % (self.task_name, self.proc_num)
    
    def run_with_combine(self, fp, data_fp, index_fp):
        result = None
        isCreated = False
        for _ in range(self.length):
            data = self.mapper.map(fp.readline())
            if data is None: continue

            if isCreated:
                result = self.mapper.combine(result, data)
            else:
                result = self.mapper.create(data)
                isCreated = True

        ShuffleIO.write_to_file(data_fp, index_fp, result)
    
    def run_without_combine(self, fp, data_fp, index_fp):
        for _ in range(self.length):
            data = self.mapper.map(fp.readline())
            if data is None: continue
            ShuffleIO.write_to_file(data_fp, index_fp, data)
    
    def run(self):
        """
            worker runs mapper function in a working loop
        """

        with open(self.infile, encoding='utf-8') as fp, \
            open(self.data_file_name, 'wb') as data_fp, \
            open(self.index_file_name, 'w') as index_fp:

            # jump to start pos
            fp.seek(self.start)

            if hasattr(self.mapper, 'combine'):
                self.run_with_combine(fp, data_fp, index_fp)
            else:
                self.run_without_combine(fp, data_fp, index_fp)
            
            self.mapper.destory()

class Master(object):
    """
        master of the parallel computing framework, create it at first
    """
    def __init__(self, infile=None, mapper=None, reducer=None, max_cores=None):
        assert mapper!=None and (type(mapper) is types.FunctionType or issubclass(type(mapper), Mapper))
        assert reducer!=None and (type(reducer) is types.FunctionType or issubclass(type(reducer), Reducer))
        assert os.path.isfile(infile)

        self.infile = infile
        self.task_name = "parallel_%d" % int(time.time())
        self.cores = cpu_count()
        
        if max_cores != None:
            assert type(max_cores) == type(1)
            assert 0 < max_cores <= cpu_count()
            self.cores = max_cores

        if type(mapper) is types.FunctionType:
            self.mapper = SimpleMapper()
            self.mapper.map = mapper
        else:
            self.mapper = mapper

        if type(reducer) is types.FunctionType:
            self.reducer = SimpleReducer()
            self.reducer.reduce = reducer
        else:
            self.reducer = reducer
    
    def run(self):
        """
            running task by mapper & reducer, using all cpu cores
        """

        starts_lengths = self.get_workload_for_workers()

        workers = []
        for i, start_len in enumerate(starts_lengths):
            proc_num = i+1
            start = start_len[0]
            length = start_len[1]

            worker = Worker(self.infile, self.task_name, proc_num, start, length, self.mapper)
            worker.proc = Process(target=lambda x:x.run(), args=(worker,))
            workers.append(worker)

        # start workers
        for worker in workers: worker.proc.start()

        # block
        self.block_until_workers_done(workers)

        # reducing
        result = self.reducing(workers)

        # join
        for worker in workers: worker.proc.join()

        return result
    
    def get_workload_for_workers(self):
        """
            get start pos & number of lines need to be mapped, for every worker
        """
        pos = 0
        total_line = 0
        line_pos = []

        # get total line and pos of every line
        with open(self.infile, encoding='utf-8') as fp:
            for line in fp:
                line_pos.append(pos)
                total_line += 1
                pos += len(line.encode('utf-8'))
        
        assert total_line > 0, "total number of lines must be more than 0"
        
        # workload(the number of lines) of every worker
        if total_line <= self.cores:
            lengths = [1 for _ in range(total_line)]
        else:
            lengths = [int(total_line/self.cores) for _ in range(self.cores)]
            for i in range(total_line%self.cores):
                lengths[i] += 1
        
        # start pos of infile for every worker
        cur_line = 0
        starts = []
        for length in lengths:
            starts.append(line_pos[cur_line])
            cur_line += length
        
        return list(zip(starts, lengths))
    
    def block_until_workers_done(self, workers):
        """
            blocking
        """
        states = [False for _ in range(len(workers))]
        while not reduce(lambda a,b: a and b, states):
            need_sleep = True

            for i, worker in enumerate(workers):
                if states[i]:
                    continue
                if not worker.proc.is_alive():
                    states[i] = True
                    need_sleep = False

            if need_sleep: time.sleep(0.2)

    def reducing(self, workers):
        """
            master process running reducer function
        """
        result = None
        isCreated = False

        # read intermediate results from files written by workers, and reduce them
        for worker in workers:
            data_file = worker.data_file_name
            index_file = worker.index_file_name

            if os.path.isfile(data_file) and os.path.isfile(index_file):
                
                # reducer
                for data in ShuffleIO.read_from_file(open(data_file, 'rb'), open(index_file, 'r')):
                    if isCreated:
                        result = self.reducer.reduce(result, data)
                    else:
                        result = self.reducer.create(data)
                        isCreated = True
                
                # delete files
                os.remove(data_file)
                os.remove(index_file)
        
        self.reducer.destory()
        return result
