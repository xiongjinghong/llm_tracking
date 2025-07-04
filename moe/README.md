# MOE
MOE_demo.ipynb 
1、basic MOE: gate 对多个 expert 加权输出
2、sparse MOE: 对gate 取top-k 个expert 加权输出, 应该有更简单的矩阵实现方式
3、SharedExpertMOE（deepseek方式）： 共享专家 和 SparseMOE 结果相加进行输出
4、损失函数：mse损失 + 0.01 * 负载均衡损失
