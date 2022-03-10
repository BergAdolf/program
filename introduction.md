<h1>叶簇隐蔽目标变化检测</h1>


**使用国内镜像源**

* 豆瓣源 
  -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

* 清华源
  -i https://pypi.tuna.tsinghua.edu.cn/simple/
  

**查看训练过程相关数据**

* 开启tensorboard
  tensorboard --logdir <your/running/dir> --port <your_bind_port>
  
* 线上查看 打开浏览器输入http://localhost:6006


**数据分析**  

train image  
* sum: 9870, train: 1974, valid: 7896 | 2:8
* pos: 3840: 3796 | 4 undetected points
* neg: 4056: 4019 | 37 false alarm points

test image  
* sum: 42757, pos: 4784, neg: 37,973
* pos: 4709 | 75   600| 589
* neg: 35567 | 2406   

