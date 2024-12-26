# pinns 系列代码使用指南
## 背景说明
pinns求解的问题如下：
$$\frac{\partial{c}}{\partial t} = \lambda^2 (\frac{\partial^2 c}{\partial x^2}+\frac{\partial^2 c }{\partial y^2}) - 5(u^3-u)$$
$$ c(x,y,0) = 0.6\cos(\pi x)\cos(\pi y)$$
$$ c(x,-1, t) = c(x, 1, t)$$
$$ c(-1, y, t) = c(1, y,t)$$
$$ c_y(x, -1, t) = c_y(x, 1,t)$$
$$ c_x(-1, y, t) = c_x(1, y, t)$$
$$ (x,y) \in [-1,1]\times[-1,1]$$
边界条件是周期性边界条件，求解二维的Allen-Cahn方程。
## 求解方法
本方法使用的物理神经网络（Physical-Informed Neural Network），神经网络各个层的厚度见程序中的layers变量，误差来源于边界条件、初始条件和偏微分方程损失。
$$ loss = loss_{pde} + loss_{initial} + loss_{boundary}$$
初始点和边界点在训练前，通过随机索引进行选取：
```python
index = np.random.choice(len(x), n, replace=False)
```
此命令从长度为len(x)的数组中返回n长度的随机索引，并且是不放回的。
而内点的选取使用拉丁超立方采样
```python
from pyDOE import lhs
train_points = lower_bound + (upper_bound - lower_bound)*lhs(3, n)
```
表示生成介于上边界和下边界之间的三维空间的点，注意这里的upper bound是三个维度上的最大值，lower bound是三个维度的最小值。
优化器使用学习率$lr = 1e-4$的adam随机梯度下降方法，实际最终训练的点数大约为46w，初始点的个数为65536(256*256)，边界点的个数大概为2w。

实际进行训练的时候我们发现很结果很有可能会在很短的时间内（实际上大约1-10个epoch）就会陷入到平凡解（c=0），对于这一点，其实也很容易预期到，因为这个平凡解除去初始值，既满足周期边界条件，也能让偏微分方程损失为0，所以随着梯度下降很快就会跌入到这个局部最优上去。

我们查阅资料(characterizing possible failure modes in physics-informed neural networks)发现这个叫做所谓的传播失败，意思是说初始条件的loss还没能逆向传播到，内部点产生了少许平凡解，于是为了降低偏微分的损失，就会有大片的空间域收敛到平凡解，通过查阅资料（solving allen-cahn and cahn-hilliard equations using the adaptive physicas informed neural networks）我们找到了一些解决办法，首先是增加初始条件损失的权重，但是这个操作实际上只能延长跌入平凡解的时间，并且还会催生新的问题，在t很小的时候会很逼近初始条件，随后会下降到平凡解。 
## 最终解决方案
我们随后意识到这是因为相场方程本身动力学的一些特性使得我们求解产生了苦难，增加反应项的系数会使得浓度变化更加剧烈并且缩短尖锐边界的时间，这种高频的信息神经网络，即便我们使用和其稳态解类似的激活函数tanh和对应的激活函数xavier权重初始化也难以刻画其趋势，于是我们使用在（solving allen-cahn and cahn-hilliard equations using the adaptive physicas informed neural networks）中提到的warm start方法。

我们首先降低反应项的数值，这个对应文件夹下面的ac_data_para1和ac_model_para_1，我们首先将初始条件的权重增加求解一个动力学相对舒缓的方程，（warm_up.py），再将模型权重保存并且赋予学习动力学更佳激烈的方程，注意的是，这里的warm up求解的模型在测试集上的数据量不需要太大。我们随后迁移到本方程的求解上面，增加增加数据量进行进一步的训练，这样可以节省训练的时间。
## 使用指南
在文件中包含有训练的数据集和保存的模型以及加载器，更改模型加载路径以后可以直接使用。
```python
path = 'path/to/ac_model_para_1 or ac_model_para_5' #改变model_load.py中加载模型的路径
python3 model_load.py #运行程序
```
也可以自己使用warm_start.py和cuda_allen_cahn_pinns.py自己训练一遍查看。

