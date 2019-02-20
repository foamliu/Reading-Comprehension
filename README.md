# 阅读理解

机器阅读理解是指让计算机阅读文本，随后让计算机解答与文中信息相关的问题。本代码库用 PyTorch 实现 DMN+ 模型,  以解决AI Challenger 2018 中的观点型问题阅读理解挑战。


## 依赖

- Python 3.6
- PyTorch 1.0

## 数据集

我们使用 AI Challenger 2018 中的阅读理解数据集。训练集25万条，验证集3万条，测试集1万条。可以在这里下载：[观点型问题阅读理解](https://challenger.ai/competition/oqmrc2018)

### 数据说明

每条数据以 <问题，篇章，候选答案> 三元组组成。每个问题对应一个篇章（500字以内），以及包含正确答案的三个候选答案。

- 问题：真实用户自然语言问题，从搜索日志中随机选取并由机器初判后人工筛选
- 篇章：与问题对应的文本段，从问题相关的网页中人工选取
- 候选答案：人工生成的答案，提供若干（三个）选项，并标注正确答案

### 数据样例

数据以JSON格式给出：
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
训练集给出上述全部字段，测试集不给answer字段。

## 用法

### 数据预处理
提取和预处理训练样本：
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