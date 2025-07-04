思路比较简单，siglip-base-patch16-224 和Qwen2.5-0.5B-Instruct进行融合

第一步、得到视觉模型的embedding
第二步、得到文本的embedding
第三步、将两个额mbedding进行拼接作为文本模型输入

问题：embedding维度不一致
解决方式：
1、文本原来<image> 部分替换成 49个特殊字符<|image_pad|>
2、视觉embedding 转换成和文本embedding一样的长度
3、进行拼接

资源说明：40G V100 上进行训练

