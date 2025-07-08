# GRPO:分组相对策略优化
## (Group Relative Policy Optimization)
## 变化1
（1）PPO:advantage 价值函数估计value,然后估算advantage \
（2）GRPO:同一个问题生成n个答案作为一组，计算每个答案reward, reward 归一化得到 advantage
## 变化2
（1）KL散度采用 r-1-log(r)的方式，有什么好处？
## loss
（1）loss = advantage * 重要性采样ratio

