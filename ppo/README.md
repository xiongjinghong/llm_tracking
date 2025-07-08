# PPO:近端策略优化（Proximal Policy Optimization, PPO）
1、sample: prompt + answer\
2、experience: \
(a)critic 得到 answer 上 values\
(b)reward 得到 rewards\
(c)计算 advantage 和 returns\
3、loss \
(a）policy_loss:  actor_model 输出，experience 中输出，advantage构建损失函数\
(b) value_loss: critic model 得到value， return 作为label \






