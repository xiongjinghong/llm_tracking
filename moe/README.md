# MOE
MOE_demo.ipynb <br/>
1、basic MOE: gate 对多个 expert 加权输出 <br/>
2、sparse MOE: 对gate 取top-k 个expert 加权输出, 应该有更简单的矩阵实现方式<br/>
3、SharedExpertMOE（deepseek方式）： 共享专家 和 SparseMOE 结果相加进行输出<br/>
4、损失函数：mse损失 + 0.01 * 负载均衡损失<br/>


问题：<br/>
1、反向传播：top_k 求导数， 只需要求完导数，将选为top_k位置的导数选中，其他位置设置为0即可<br/>