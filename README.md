# 观点型问题阅读理解

机器阅读理解是指让计算机阅读文本，随后让计算机解答与文中信息相关的问题。这里重点针对阅读理解中较为复杂的，需要利用整篇文章中多个句子的信息进行综合才能得到正确答案的观点型问题。


## 依赖

- Python 3.6
- PyTorch 1.0

## 数据集

我们使用 AI Challenger 2018 中的观点型问题阅读理解数据集，超过1000万的英中对照的句子对作为数据集合。其中，训练集合占据绝大部分，验证集合8000对，测试集A 8000条，测试集B 8000条。

可以从这里下载：[观点型问题阅读理解](https://challenger.ai/competition/oqmrc2018)

数据说明
每条数据为<问题，篇章，候选答案> 三元组组成

每个问题对应一个篇章（500字以内），以及包含正确答案的三个候选答案

问题：真实用户自然语言问题，从搜索日志中随机选取并由机器初判后人工筛选

篇章：与问题对应的文本段，从问题相关的网页中人工选取

候选答案：人工生成的答案，提供若干（三个）选项，并标注正确答案

数据以JSON格式表示如下样例：
<pre>
{
“query_id”:1,
“query”:“维生c可以长期吃吗”,
“url”: “https://wenwen.sogou.com/z/q748559425.htm”,
“passage”: “每天吃的维生素的量没有超过推荐量的话是没有太大问题的。”,
“alternatives”:”可以|不可以|无法确定”,
“answer”:“可以”
}
</pre>
训练集给出上述全部字段，测试集不给answer字段

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