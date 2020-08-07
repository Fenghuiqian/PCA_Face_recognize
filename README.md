# PCA_Face_recognize
手写实现PCA和LDA做人脸识别，比较其效果
# 基本步骤
## PCA
1. 用协方差矩阵的方法求解训练集的特征值、特征向量。
2. 查看特征值曲线，以提取80%以上特征的原则，确定特征向量的个数。
3. 将每个训练集与测试集的图像矩阵与特征向量相乘进行降维，再flatten成一维向量。
4. 用KNN，其中K=1的最近邻的方法，对测试集的每个一维向量与训练集的向量进行点乘操作，其中最大的值对应的训练集的标签，即为测试集的预测标签。
## LDA
1. 构造类间散步矩阵和类内散步矩阵，再计算训练集的特征值、特征向量。
2. 之后步骤与PCA相同。
# Notes
LDA的效果稍好于PCA。
显然，LDA寻找特征时考虑了标签的信息，更类似于有监督训练，在投影后不同类别的数据点的距离最大化，使得特征在进行这种分类时，更加有效。
