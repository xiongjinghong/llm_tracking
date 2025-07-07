# 知识蒸馏
1、将教师模型输出和学生模型输出计算kl距离，当成loss <br />
2、继承Trainer,实现 compute_loss(self, model, inputs, return_outputs=False) 方法即可 <br />