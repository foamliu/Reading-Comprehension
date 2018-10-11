# 观点型问题阅读理解

机器阅读理解是指让计算机阅读文本，随后让计算机解答与文中信息相关的问题。这里重点针对阅读理解中较为复杂的，需要利用整篇文章中多个句子的信息进行综合才能得到正确答案的观点型问题。


## 依赖

- Python 3.5
- PyTorch 0.4

## 数据集

我们使用AI Challenger 2018 中的观点型问题阅读理解数据集，超过1000万的英中对照的句子对作为数据集合。其中，训练集合占据绝大部分，验证集合8000对，测试集A 8000条，测试集B 8000条。

可以从这里下载：[英中翻译数据集](https://challenger.ai/competition/oqmrc2018)

![image](https://github.com/foamliu/Machine-Translation/raw/master/images/dataset.png)

## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python extract.py
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

### Demo
下载 [预训练模型](https://github.com/foamliu/Machine-Translation/releases/download/v1.0/BEST_checkpoint.tar) 放在 models 目录然后执行:

```bash
$ python demo.py
```

<pre>


</pre>