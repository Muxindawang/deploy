

## 并行处理简介

- 串行：指令或者代码块必须是依次执行。**针对于复杂逻辑运算。**

- 并行：指令或代码块同时执行，利用多核的特性去完成一个或多个任务。**擅长一些大规模的科学计算，如天体预测、图像处理等。**

  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4fc1387175c44730a7e6312e71a33103.png#pic_center)

  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a807269fd27145dc80fc4e671767264d.png#pic_center)

并发的意思是我们只有一个 CPU，我们要让它执行两个 task，先让它执行 task1，当 task1 执行到一半时让它去执行 task2，task2 做完一半后继续回到之前做 task1，之后再做 task2，如此反复循环交替。当然实际上这个切换 task1 和 task2 的速度非常非常快，人眼是无法识别的，所以我们能够感觉到这个 CPU 它是同时在做两个任务，那这个就是并发，逻辑意义上的同时执行。
并行它就是**物理意义上的同时执行**，比如我们现在有两个 CPU，我们让这两个 CPU 分别处理 task1 和 task2，并且同时去执行，那这个就是并行



- SIMD 全称 **S**ingle **I**nstruction **M**ultiple **D**ata，也就是同一条指令去执行多个数据。通俗的讲就是，有若干个相同逻辑的计算过程，我们需要先读取数据、做计算、再写入数据；SIMD operation可以一次性读取全部的数据，然后只进行一次计算，写入一次数据即可。

![img](https://i-blog.csdnimg.cn/direct/9c2b1e4511b2456e8516b407134873f3.png#pic_center)

CPU 和 GPU 在并行处理上的优化方向并不相同，CPU 的目标在于如何去最大化的减少 memory latency 访问，访问 memory 的一个延迟时间；而 GPU 的优化目标在于如何去提高程序的 throughout

CPU 并行处理优化方式主要有：Pipeline、cache hierarchy、Pre-fetch、Branch-prediction、Multi-threading 等

GPU 并行处理优化方式主要有：multi-threading、warp schedule

## 参考

https://blog.csdn.net/qq_40672115/article/details/143697224