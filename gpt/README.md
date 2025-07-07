# gpt简化版本
知识点：
1、位置编码
positional Embedding位置编码:
PE(pos, 2i) = sin(pos/10000^(2i)/dim)
dim:词向量长度
pos是token位置
2i是词向量2i位置
公式是说 第pos个token对应第词向量的2i位置对应的值为sin(pos/10000^(2i)/dim)
PE(pos, 2i+1) = cos(pos/10000^(2i)/dim)

Rotary Position Embedding
二维向量可以利用旋转矩阵，
旋转矩阵实现将第m个向量v_m旋转 m*theta角度, 旋转对应矩阵为P，这里变换后为P^m v_m
旋转矩阵实现将第n个向量v_n 旋转 n*theta角度, 旋转对应矩阵为P，这里变换后为P^n v_n
两矩阵内积 为v_m.T * p.T^m * P^n v_n = v_m * p^(n-m)v_n
这里利用对于旋转矩阵，就是正交矩阵，满足 p.T* P = 单位矩阵, 如果p代表旋转一次，p^m 就是旋转m次
所以这里第m个位置token embedding, 加入位置embedding的方式就是将原来向量旋转m*theta

2、RMSNorm
LayerNorm: 计算均值和方差，然后归一化
归一化只计算方差，不计算均值https://blog.csdn.net/2301_79093491/article/details/143437326

3、group query attention:
https://blog.csdn.net/shizheng_Li/article/details/145831357
查询向量（Query, Q） 维度：(B , L , H_q , D) 
键向量（Key, K） 维度：(B , L , H_k , D) 
值向量（Value, V） 维度：(B , L , H_k , D) 
Q 维度不变， KV从 num_heads 变成 num_kv_heads, 然后进行复制
原文链接：https://blog.csdn.net/shizheng_Li/article/details/145831357

4、数据构建
pretrain: input_ids:text_idx[:-1]   labels:text_idx[1:] 
sft: input_ids: (q_idx + a_idx)[:-1]  labels:(pad补充 + a_idx)[1:]






