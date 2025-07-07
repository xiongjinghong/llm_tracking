# DPO
损失说明：
dpo 直接偏好优化的实现
1、Bradley-Terry能力模型假设计算reward
2、reward最大化+KL散度约束得到 reward 形式
3、正样本优于负样本极大似然进一步化简
得到简化后的损失函数
logits = logratios - ref_logratios = （chosen_logps - rejected_logps） - （ref_chosen_logps - ref_rejected_logps）
通俗理解，模型对数似然应该满足：1、接受样本的大于拒绝样本，用差作指标，这个差值越大越好  2、新训练模型的差 - 原始模型的差越大越好

样本构建：
input = [query + accepted[:-1], ... query + rejected[:-1],...]
label = [pad + accepted[1:], ... pad + rejected[1:],...]
相当于原始1个batch里面两条回答，构建出的数据变成了2*batch_size 大小