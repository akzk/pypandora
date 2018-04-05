# README

**pypandora**是一个为了方便数据科学工作的python工具库。目前，只开源其中一个模块。

* parallet

## parallet

**pypandora.parallet**是一个单机多核并行框架. 由于PIL的限制，CPython的多线程只能满足并发要求，无法提供多核并行的功能。在一部多核心的机器上，为了提高核心利用率、减少任务运行时间，pypandora.parallet利用多进程方式提供多核并行服务。pypandora.parallet的开发灵感源于Hadoop的MapReduce计算框架，但是对其框架进行了简化与修改，例如MapReduce中不管是Mapper还是Reducer需要花费部分算力在排序上面，而pypandora.parallet则删除了这部分；MapReduce中reduce处理函数接收的是一个形如(key, [value1, value2, ...])的参数，意为众Mapper产生的同key的value序列，而pypandora.parallet中的reduce与标准库functools.reduce的用法一致。因此，pypandora.parallet内部实现的许多概念不能与MapReduce等同。下面是统计词频的例子。

### 用例

1. 函数

   最简单的做法就是直接定义map、reduce两个函数

   ```python
   import pypandora.parallet as parallet

   def simple_mapper(line):
       words = line.split()
       word_frq = {}
       for word in words:
           if word in word_frq:
               word_frq[word] += 1
           else:
               word_frq[word] = 1
       return word_frq

   def simple_reducer(a, b):
       for k, v in b.items():
           if k in a:
               a[k] += v
           else:
               a[k] = v
       return a

   print(parallet.run("path/to/file", simple_mapper, simple_reducer))
   ```

2. 类

   上述直接定义函数的做法在某些任务上，会由于Mapper产生中间文件时的IO频繁导致运行过慢，需要引入combine减少中间结果的数据量；有时，自定义的map、reduce函数需要用到其他参数；有时，某些模块被加载入父进程后，在开启子进程时会报错，需要在子进程启动时才加载。所以，这里提供定义类来运行框架的方式

   ```python
   class SimpleMapper(parallet.Mapper):
       def import_modules(self):
           pass
       def map(self, line):
           return simple_mapper(line)
       def combine(self, a, b):
           return simple_reducer(a, b)

   class SimpleReducer(parallet.Reducer):
       def import_modules(self):
           pass
       def reduce(self, a, b):
           return simple_reducer(a, b)

   print(parallet.run("path/to/file", SimpleMapper(), SimpleReducer()))
   ```

### 测试1

为了得知该框架的性能，统计在不同核心数情况下wordcount任务的时间，并与没有使用框架的用时进行对比

> * 环境
>
> 系统：OS X EI Capitan，10.11.6
>
> CPU：2.2 GHz Intel Core i7，四核心八线程
>
> 内存：16 GB 1600 MHz DDR3
>
> 磁盘：APPLE SSD SM0256G
>
> Python版本：3.6.2
>
> 测试文件：《飘》英文版（扩充40倍，约100M）

![wordcount](img/parallet_wordcount.png)

### 测试2

实际场景中的任务——对语料进行中文分词，属于计算密集型任务，且行与行之间的结果相互无影响。

> * 环境
>
> 测试文件：语料文件（约30M，一行一篇中文文章）
>
> 其余条件同上

![jieba](img/parallet_jieba.png)