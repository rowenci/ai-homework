# ai-homework

使用卷积神经元网络CNN，对多种字体的26个大写英文字母进行识别。

数据集介绍：
（1）、数据集来源于Chars74K dataset，本项目选用数据集EnglishFnt中的一部分。Chars74K dataset网址链接 http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/；
（2）、A-Z共26种英文字母，每种字母对应一个文件夹（Sample011对应字母A, Sample012对应字母B,…, Sample036对应字母Z）；
（3）、Sample011到Sample036每个文件夹下相同字母不同字体的图片约1000张，PNG格式；
（4）、本项目数据集请从以下链接下载:
https://pan.baidu.com/s/1HEsbvusyYCni7MVGKUk4bA， 提取码：dhix

要求：

1. 每种字母当成一类，利用卷积神经元网络构建26类分类器；
2. 每个类别中随机选择80%作为训练数据集，剩余20%作为测试数据集。采用训练集进行模型训练，采用测试集进行模型测试，并给出测试集准确率结果。

Bonus:
1、	Bonus文件夹下为手写A-Z的字母图片。请将之前训练好的分类器迁移学习到Bonus数据集上，重新构建分类器，Bonus数据集中随机选择80%作为训练数据集，剩余20%作为测试数据集，并给出测试集准确率结果。
2、	将Bonus文件夹下的图片当作未标注类别的数据，联合之前的标注图片，采用半监督学习的方法构建分类器。
3、	其它。
