# CS336 作业 1（基础）：构建 Transformer 语言模型

> 版本 26.0.3｜CS336 课程团队｜2026 年春季  
> 本文是 `cs336_assignment1_basics.pdf` 的非官方中文翻译。公式、代码标识、测试命令和交付要求尽量保持原样；如有歧义，请以英文原文为准。

## 1 作业概览

在本次作业中，你将从零构建训练标准 Transformer 语言模型（LM）所需的全部组件，并训练若干模型。

### 你将实现的内容

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）
4. 训练循环，支持序列化及加载模型与优化器状态（第 5 节）

### 你将运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器编码数据集，将其转换成整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度。
5. 在 OpenWebText 上训练模型，并将达到的困惑度提交到排行榜。

### 可以使用的内容

我们希望你从零构建每个组件。特别是，除了以下项目以外，不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类，例如 `Module`、`ModuleList`、`Sequential` 等[^1]
- `torch.optim.Optimizer` 基类

你可以使用 PyTorch 的其他任何定义。如果想使用某个函数或类，却不确定是否允许，请随时在 Slack 上提问。拿不准时，请考虑使用它是否违背本作业“从零实现”的理念。

### 关于 AI 工具的声明

AI 能够完全自主地解决作业中的许多部分，这会使学生更难深入参与课程内容并从中学习。

允许使用 AI 工具回答高层次的概念问题，或提供函数签名、库 API 等底层编程文档。但是，不允许使用 AI 工具实现任何作业的任何部分。这既包括编程智能体（例如 Cursor Agents、Codex、Claude Code），也包括 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。使用 AI 智能体时，请确保它使用所提供的 `AGENTS.md` 文件。使用聊天机器人时，还应一并提交提示词。

我们强烈建议你在完成作业时关闭 IDE 中的 AI 自动补全（例如 Cursor Tab、GitHub Copilot）；非 AI 自动补全（如补全函数名）完全没有问题。往届学生指出，关闭 AI 自动补全后更容易深入投入课程材料。

完整的 AI 政策请参阅原文链接中的相应文档。

### 代码仓库的结构

作业代码和本讲义位于 GitHub：

<https://github.com/stanford-cs336/assignment1-basics>

请克隆该仓库。如果有更新，课程团队会通知你，以便通过 `git pull` 获取最新版本。

1. `cs336_basics/*`：在这里编写代码。注意，这里没有预置代码——你可以完全从零开始，自由组织实现。
2. `adapters.py`：你的代码必须提供一组指定功能。对于每项功能（例如缩放点积注意力），只需通过调用自己的代码来填写其适配器实现（例如 `run_scaled_dot_product_attention`）。注意：对 `adapters.py` 的修改不应包含任何实质逻辑；它只是胶水代码。
3. `test_*.py`：这里包含必须通过的全部测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

### 如何提交

运行 `make_submission.sh` 生成提交用的 zip 文件。如果你有不希望放入提交包的大型数据文件或检查点，请务必把它们加入脚本的排除列表。

你需要向 Gradescope 提交：

- `writeup.pdf`：回答所有书面问题。请使用排版工具整理答案。
- `code.zip`：包含你编写的全部代码。

如需提交排行榜，请向以下仓库提交 PR：

<https://github.com/stanford-cs336/assignment1-basics-leaderboard>

详细说明请参阅排行榜仓库中的 `README.md`。

### 从哪里获取数据集

本作业使用两个预处理数据集：TinyStories [R. Eldan 等，2023] 和 OpenWebText [A. Gokaslan 等，2019]。二者都是单个大型纯文本文件。

如果你随班修读，请在计算资源指南中查看数据集下载说明。如果你在家自学，可以使用 `README.md` 中的命令下载这些文件。

> **低资源提示：说明**  
> 在整门课程的作业讲义中，我们会提供使用较少 GPU 资源、甚至不使用 GPU 完成某些部分的建议。例如，有时会建议缩小数据集或模型规模，或说明如何在 Mac 集成 GPU 或 CPU 上运行训练代码。这类“低资源提示”会像本段一样出现在蓝色框中。即使你是拥有课程机器访问权限的 Stanford 在读学生，这些提示也可能帮助你加快迭代并节省时间，因此建议阅读。

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**  
> 使用课程团队的参考实现，我们可以在配有 36 GB 内存的 Apple M4 Max 芯片上训练一个能够生成较流畅文本的 LM：使用 Metal GPU（MPS）不到 5 分钟，使用 CPU 约 30 分钟。如果你不熟悉这些术语也无需担心；只要你的笔记本比较新、实现正确且高效，就能够训练一个小型 LM，生成流畅度尚可的简单儿童故事。后文将说明在 CPU 或 MPS 上需要进行哪些调整。

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器 [R. Sennrich 等，2016；C. Wang 等，2019]。具体而言，我们会把任意 Unicode 字符串表示为字节序列，并在该序列上训练 BPE 分词器。之后，我们会用它把文本（字符串）编码成用于语言建模的 token（整数序列）。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它把字符映射到整数码点。截至 2025 年 9 月发布的 Unicode 17.0，该标准在 172 种文字系统中定义了 159,801 个字符。例如，字符 `s` 的码点为 115（通常写作 U+0073，其中 `U+` 是惯用前缀，0073 是 115 的十六进制表示），字符“牛”的码点为 29275。

在 Python 中，可以用 `ord()` 将单个 Unicode 字符转换成整数表示；`chr()` 则把整数 Unicode 码点转换成相应字符组成的字符串。

```pycon
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

#### 问题（unicode1）：理解 Unicode（1 分）

**(a)** `chr(0)` 返回哪个 Unicode 字符？

**交付内容：** 一句话回答。

**(b)** 这个字符的字符串表示（`__repr__()`）与打印出来的表示有何不同？

**交付内容：** 一句话回答。

**(c)** 当这个字符出现在文本中时会发生什么？可以在 Python 解释器中尝试以下内容，观察结果是否符合预期：

```pycon
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**交付内容：** 一句话回答。

### 2.2 Unicode 编码

Unicode 标准定义了字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不实际：词表会大得难以承受（约 15 万项），而且非常稀疏，因为许多字符极少出现。因此，我们改用 Unicode 编码，把一个 Unicode 字符转换成字节序列。Unicode 标准本身定义了 UTF-8、UTF-16 和 UTF-32 三种编码，其中 UTF-8 是互联网的主导编码（超过 98% 的网页使用它）。

在 Python 中，可以用 `encode()` 将 Unicode 字符串编码为 UTF-8。要访问 Python `bytes` 对象底层的字节值，可以迭代它（例如调用 `list()`）。最后，可以用 `decode()` 将 UTF-8 字节串解码为 Unicode 字符串。

```pycon
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串中的字节值（0 到 255 的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

把 Unicode 码点转换为字节序列（例如使用 UTF-8），本质上就是把码点序列（21 位整数，其中有 159,801 个有效值）变换成字节值序列（0 到 255 的整数）。长度为 256 的字节词表更容易处理。采用字节级分词后，我们无需担心词表外 token，因为任何输入文本都可以表示成 0 到 255 的整数序列。

#### 问题（unicode2）：Unicode 编码（3 分）

**(a)** 与 UTF-16 或 UTF-32 相比，选择在 UTF-8 编码的字节上训练分词器有哪些理由？比较不同输入字符串在这些编码下的输出可能会有帮助。

**交付内容：** 一到两句话回答。

**(b)** 考虑下面这个错误的函数。它原本想把 UTF-8 字节串解码成 Unicode 字符串。该函数为什么不正确？请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**交付内容：** 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串，并用一句话解释该函数为什么错误。

**(c)** 给出一个无法解码为任何 Unicode 字符的双字节序列。

**交付内容：** 一个示例及一句话解释。

### 2.3 子词分词

字节级分词可以缓解词级分词器的词表外问题，但把文本分成字节会产生极长的输入序列。这会拖慢模型训练：一个含 10 个单词的句子，在词级语言模型中可能只有 10 个 token，但在字符级模型中可能有 50 个甚至更多 token（取决于单词长度）。处理更长的序列会使模型每一步都需要更多计算；此外，较长的输入序列会在数据中形成长期依赖，使字节序列上的语言建模更加困难。

子词分词位于词级与字节级分词之间。字节级分词器的词表有 256 项（字节值为 0 到 255）；子词分词器通过增大词表来换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在原始训练文本中频繁出现，将它作为一个词表项，就能把原来的 3-token 序列缩减成单个 token。

如何选择加入词表的子词单元？R. Sennrich 等人提出采用字节对编码（BPE；P. Gage），这是一种压缩算法：反复把最常见的字节对替换（“合并”）为一个新的、尚未使用的索引。该算法会向词表加入能最大限度压缩输入序列的子词 token；如果某个词在输入文本中出现得足够多，它就会被表示成一个子词单元。

通过 BPE 构造词表的子词分词器通常称为 BPE 分词器。本作业将实现字节级 BPE 分词器，其词表项是单个字节或合并后的字节序列。这样既能妥善处理词表外输入，又能让输入序列长度保持在可管理范围。构造 BPE 分词器词表的过程称为“训练”BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤。

#### 初始化词表

分词器词表是从字节串 token 到整数 ID 的一一映射。由于训练的是字节级 BPE 分词器，初始词表就是全部字节的集合。字节共有 256 种可能取值，因此初始词表大小为 256。

#### 预分词

有了词表以后，原则上可以统计文本中相邻字节的出现次数，并从频率最高的字节对开始合并。但这种做法计算开销很大，因为每次合并都要完整遍历语料库。此外，直接跨整个语料合并字节可能产生仅标点不同的 token，例如 `dog!` 与 `dog.`。虽然二者很可能语义高度相似，却会获得完全不同的 token ID。

为避免这些问题，我们先对语料进行预分词。可以把它理解为对语料进行一次粗粒度分词，帮助统计相邻字符出现的频率。例如，单词 `text` 可能是一个出现 10 次的预 token。统计 `t` 与 `e` 的相邻次数时，我们知道 `text` 中二者相邻，于是可以一次把计数加 10，而无需逐处扫描语料。由于训练的是字节级 BPE 模型，每个预 token 都表示为 UTF-8 字节序列。

R. Sennrich 等人的原始 BPE 实现只是按空白进行预分词（即 `s.split(" ")`）。基于 SentencePiece 的分词器仍可见这种方法，例如 Llama 1 和 2 的分词器。

大多数现代分词器使用基于正则表达式的预分词器，这一实践来自 GPT-2 [A. Radford 等]。我们将使用原始正则表达式稍加整理后的版本，取自：

<https://github.com/openai/tiktoken/pull/234/files>

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

可以交互式地用该预分词器拆分一些文本，以便理解它的行为：

```pycon
>>> # 需要安装 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在实际代码中应使用 `re.finditer`，以免在构建预 token 及其计数映射时把所有预分词结果都存入内存。

#### 计算 BPE 合并

把输入文本转换成预 token，并将每个预 token 表示成 UTF-8 字节序列后，就可以计算 BPE 合并，即训练 BPE 分词器。从高层来看，BPE 算法反复统计每个字节对，并找出频率最高的字节对 `(A, B)`；随后合并该字节对的每次出现，即用新 token `AB` 替换它。新合并的 token 会加入词表。因此，BPE 训练完成后的词表大小等于初始词表大小（本例为 256）加训练过程中执行的 BPE 合并次数。

为了提高训练效率，我们不考虑跨越预 token 边界的字节对。[^2] 当多个字节对频率相同时，确定性地选择字典序更大的字节对。例如，如果 `(A, B)`、`(A, C)`、`(B, ZZ)` 和 `(BA, A)` 的频率并列最高，应合并 `(BA, A)`：

```pycon
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### 特殊 token

某些字符串（例如 `<|endoftext|>`）通常用于编码元数据（如文档边界）。编码文本时，往往希望把某些字符串视为“特殊 token”，永远不把它们拆成多个 token。例如，序列结束字符串 `<|endoftext|>` 应始终保留为单个 token（即单个整数 ID），从而让我们知道何时停止语言模型生成。这些特殊 token 必须加入词表，才能拥有对应的固定 token ID。

R. Sennrich 等人论文的算法 1 给出了一个低效的 BPE 分词器训练实现，基本遵循上面列出的步骤。作为第一个练习，实现并测试该函数可能有助于检查自己的理解。

#### 示例（bpe_example）：BPE 训练示例

下面是改编自 R. Sennrich 等人的一个示意性例子。假设语料由以下文本组成：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词表中有特殊 token `<|endoftext|>`。

**词表。** 用特殊 token `<|endoftext|>` 和全部 256 个字节值初始化词表。

**预分词。** 为简化问题并聚焦合并过程，本例假设预分词只是按空白拆分。预分词并计数后得到频率表：

```text
{low: 5, lower: 2, widest: 3, newest: 6}
```

可以方便地将其表示为 `dict[tuple[bytes, ...], int]`，例如 `{(l,o,w): 5, …}`。注意，在 Python 中，即便单个字节也是一个 `bytes` 对象。Python 不存在用于表示单个字节的 `byte` 类型，正如它没有用于表示单个字符的 `char` 类型。

**合并。** 首先查看每一对连续字节，并对包含该字节对的词的频率求和：

```text
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}
```

字节对 `(e, s)` 与 `(s, t)` 并列，因此选择字典序更大的 `(s, t)`。合并预 token 后得到：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}
```

第二轮中，`(e, st)` 是最常见的字节对（计数为 9），合并后得到：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}
```

继续执行，最终合并序列为：

```text
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west',
 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
```

若只执行 6 次合并，则合并序列为 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，词表元素为 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。使用该词表和合并集合，单词 `newest` 会被分为 `[ne, west]`。

### 2.5 BPE 分词器训练实验

现在在 TinyStories 数据集上训练一个字节级 BPE 分词器。数据集查找与下载说明见第 1 节。开始前，建议先浏览 TinyStories 数据集，了解其中的内容。

#### 并行化预分词

你会发现预分词步骤是一个主要瓶颈。可以使用内置库 `multiprocessing` 并行化代码，从而加速预分词。具体来说，在并行实现中，建议对语料分块，并确保块边界位于特殊 token 的起始位置。你可以原样使用以下链接中的起始代码来获取块边界，再把任务分配给多个进程：

<https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py>

这种分块方式总是有效，因为我们绝不希望跨文档边界合并。就本作业而言，可以始终用这种方式拆分，无需考虑语料极大且不含 `<|endoftext|>` 的边界情况。

#### 预分词前移除特殊 token

使用正则模式运行预分词（通过 `re.finditer`）之前，应从语料中移除所有特殊 token；并行实现中则从当前块移除。请确保按特殊 token 拆分，以免合并跨越它们所划定的文本边界。

例如，若语料或块为 `[Doc 1]<|endoftext|>[Doc 2]`，应按 `<|endoftext|>` 拆分，并分别对 `[Doc 1]` 与 `[Doc 2]` 预分词，防止跨文档边界合并。换言之，特殊 token 在训练期间定义了硬分段边界，但不应参与合并计数。可以使用 `re.split`，以 `"|".join(special_tokens)` 作为分隔符；因为特殊 token 中可能包含 `|`，需谨慎使用 `re.escape`。测试 `test_train_bpe_special_tokens` 会检查这一点。

#### 优化合并步骤

上述示意例子中的朴素 BPE 训练实现速度很慢，因为每次合并都要遍历全部字节对，找出频率最高者。然而，每次合并后，只有与被合并字节对重叠的字节对计数会变化。因此，可以索引所有字节对的计数并增量更新，而不是每次都显式遍历所有字节对重新统计频率。这种缓存方法能带来显著加速；但应注意，Python 中 BPE 训练的合并部分无法并行化。

> **低资源提示：性能分析**  
> 应使用 `cProfile` 或 `py-spy` 等性能分析工具找出实现中的瓶颈，并集中优化这些部分。

> **低资源提示：“缩小规模”**  
> 不要一上来就在完整 TinyStories 数据集上训练分词器。建议先用一小部分数据，即“调试数据集”；例如可以改用 TinyStories 验证集，其中有 2.2 万篇文档，而不是 212 万篇。这体现了一种通用策略：尽可能缩小数据集、模型等规模以加快开发。选择调试数据集大小或超参数配置时需谨慎：它应足够大，以体现完整配置中的同类瓶颈，使优化能够泛化；但又不能大到每次运行都耗时过久。

#### 问题（train_bpe）：训练 BPE 分词器（15 分）

**交付内容：** 编写一个函数，给定输入文本文件路径，训练字节级 BPE 分词器。该函数至少应处理以下输入参数：

| 输入 | 类型 | 说明 |
|---|---|---|
| `input_path` | `str` | BPE 分词器训练数据文本文件的路径。 |
| `vocab_size` | `int` | 正整数，定义最终词表的最大大小，包括初始字节词表、合并产生的词表项以及全部特殊 token。 |
| `special_tokens` | `list[str]` | 要加入词表的字符串列表。训练期间把它们视为硬边界，阻止跨越其范围进行合并，但计算合并统计量时不包含它们。 |

训练函数应返回最终词表和合并列表：

| 输出 | 类型 | 说明 |
|---|---|---|
| `vocab` | `dict[int, bytes]` | 分词器词表，从整数 token ID 映射到 token 字节。 |
| `merges` | `list[tuple[bytes, bytes]]` | 训练产生的 BPE 合并列表。每项是字节元组 `(<token1>, <token2>)`，表示把 `<token1>` 与 `<token2>` 合并；列表应按创建顺序排列。 |

要使用课程提供的测试验证训练函数，需先实现测试适配器 `adapters.run_train_bpe`，然后运行：

```shell
uv run pytest tests/test_train_bpe.py
```

你的实现应能通过全部测试。作为可选项（可能需要投入大量时间），可以用某种系统语言实现训练方法的关键部分，例如 C++（可考虑 `cppyy` 或 `nanobind`）或 Rust（使用 PyO3）。如果这样做，请留意哪些操作需要复制 Python 内存、哪些能够直接读取，并提供构建说明，或确保只使用 `pyproject.toml` 即可完成构建。还要注意，大多数正则表达式引擎都不能很好地支持 GPT-2 正则表达式，即使支持也往往过慢。课程团队已验证 Oniguruma 速度尚可且支持负向前瞻，但 Python 的 `regex` 包实际上可能更快。

#### 问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）

**(a)** 在 TinyStories 数据集上训练字节级 BPE 分词器，最大词表大小为 10,000。务必把 TinyStories 的特殊 token `<|endoftext|>` 加入词表。将最终词表和合并列表序列化到磁盘，以便进一步检查。训练耗费了多少时间和内存？词表中最长的 token 是什么？它合理吗？

**资源要求：** 不超过 30 分钟（不使用 GPU），内存不超过 30 GB。

> **提示** 通过在预分词阶段使用多进程，并利用以下两个事实，BPE 训练应能控制在 2 分钟以内：  
> (a) `<|endoftext|>` token 在数据文件中划分文档；  
> (b) 在应用 BPE 合并之前，`<|endoftext|>` token 已作为特殊情况处理。

**交付内容：** 一到两句话回答。

**(b)** 对代码进行性能分析。分词器训练流程的哪一部分耗时最多？

**交付内容：** 一到两句话回答。

接下来尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。与之前一样，建议先浏览数据集，以便理解其内容。

#### 问题（train_bpe_expts_owt）：在 OpenWebText 上进行 BPE 训练实验（2 分）

**(a)** 在 OpenWebText 数据集上训练字节级 BPE 分词器，最大词表大小为 32,000。将最终词表和合并列表序列化到磁盘，以便进一步检查。词表中最长的 token 是什么？它合理吗？

**资源要求：** 不超过 12 小时（不使用 GPU），内存不超过 100 GB。

**交付内容：** 一到两句话回答。

**(b)** 比较在 TinyStories 与 OpenWebText 上训练得到的分词器，说明二者的异同。

**交付内容：** 一到两句话回答。

### 2.6 BPE 分词器：编码与解码

上一部分中，我们实现了一个在输入文本上训练 BPE 分词器的函数，得到了分词器词表和 BPE 合并列表。现在要实现一个 BPE 分词器：它加载给定词表与合并列表，并用它们在文本与 token ID 之间进行编码和解码。

#### 2.6.1 编码文本

BPE 文本编码过程与 BPE 词表训练相呼应，包含几个主要步骤。

**第 1 步：预分词。** 与 BPE 训练时一样，先对序列进行预分词，并把每个预 token 表示成 UTF-8 字节序列。随后在每个预 token 内部把这些字节合并成词表元素；每个预 token 独立处理，不跨预 token 边界合并。

**第 2 步：应用合并。** 按照 BPE 训练时的创建顺序，依次把词表元素的合并应用到预 token。

##### 示例（bpe_encoding）：BPE 编码示例

假设输入字符串为 `'the cat ate'`，词表为：

```text
{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
```

学习到的合并列表为：

```text
[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
```

预分词器先把字符串拆成 `['the', ' cat', ' ate']`，然后对每个预 token 应用 BPE 合并。

第一个预 token `'the'` 起初表示为 `[b't', b'h', b'e']`。查看合并列表，首个适用的合并是 `(b't', b'h')`，将其变成 `[b'th', b'e']`。随后重新查看合并列表，找到下一个适用的 `(b'th', b'e')`，将其变成 `[b'the']`。此时没有其他合并可再应用，因此相应整数序列为 `[9]`。

对其余预 token 重复该过程：`' cat'` 应用 BPE 合并后表示成 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`；最后一个预 token `' ate'` 变成 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，输入字符串的最终编码结果为 `[9, 7, 1, 5, 10, 3]`。

#### 特殊 token

编码文本时，分词器应能正确处理构造分词器时提供的、由用户定义的特殊 token。

#### 内存注意事项

假设要对一个无法装入内存的大型文本文件进行分词。为了高效处理这个大型文件或任意数据流，需要把它拆成大小可管理的块，并依次处理各块，使内存复杂度保持常数，而不是随文本大小线性增长。分块时必须确保 token 不跨越块边界，否则结果会不同于把整个序列一次性载入内存进行分词的朴素方法。

#### 2.6.2 解码文本

要把整数 token ID 序列解码回原始文本，只需在词表中查出每个 ID 对应的字节序列，把它们连接起来，再将字节解码为 Unicode 字符串。

注意，输入 ID 不保证能映射成有效的 Unicode 字符串，因为用户可以输入任意整数 ID 序列。如果输入 token ID 没有产生有效 Unicode 字符串，应把格式错误的字节替换为 Unicode 官方替换字符 U+FFFD。[^3] `bytes.decode` 的 `errors` 参数控制 Unicode 解码错误的处理方式；使用 `errors='replace'` 会自动用替换标记代替格式错误的数据。

#### 问题（tokenizer）：实现分词器（15 分）

**交付内容：** 实现一个 `Tokenizer` 类。给定词表和合并列表，它能够把文本编码成整数 ID，并把整数 ID 解码成文本。分词器还应支持用户提供的特殊 token；如果这些 token 尚不在词表中，则把它们追加到词表。建议采用下文给出的接口。

[^1]: 完整列表见 <https://pytorch.org/docs/stable/nn.html#containers>。
[^2]: 原始 BPE 方案指定加入词尾 token。训练字节级 BPE 模型时不加入词尾 token，因为包括空白和标点在内的所有字节都已进入模型词表。由于显式表示了空白和标点，学到的 BPE 合并自然会反映这些词边界。
[^3]: 关于 Unicode 替换字符的更多信息，见 <https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character>。

建议接口如下：

- `__init__(self, vocab, merges, special_tokens=None)`：用 `vocab: dict[int, bytes]`、`merges: list[tuple[bytes, bytes]]` 和可选的 `special_tokens: list[str] | None` 构造分词器。
- `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`：类方法，从序列化词表、合并列表及可选特殊 token 构造并返回 `Tokenizer`。
- `encode(self, text: str) -> list[int]`：把文本编码为 token ID 序列。
- `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定字符串可迭代对象（例如文件句柄），惰性地产生 token ID，以便按恒定内存处理大型文件。
- `decode(self, ids: list[int]) -> str`：把 token ID 序列解码为文本。

实现 `adapters.get_tokenizer`，然后运行：

```shell
uv run pytest tests/test_tokenizer.py
```

实现应通过全部测试。

### 2.7 实验

#### 问题（tokenizer_experiments）：分词器实验（4 分）

**(a)** 分别从 TinyStories 和 OpenWebText 抽取 10 篇文档，用此前训练的 10K TinyStories 分词器和 32K OpenWebText 分词器编码。各自的压缩率（字节/token）是多少？

**(b)** 用 TinyStories 分词器处理 OpenWebText 样本会发生什么？比较压缩率和/或定性描述结果。

**(c)** 估算分词器吞吐量（如字节/秒）。对 825 GB 文本的 Pile 数据集分词需要多久？

**(d)** 用两个分词器分别编码对应的训练集和开发集，供后续训练语言模型。建议把 token ID 序列化为 `uint16` NumPy 数组。为什么 `uint16` 合适？

以上各小题的**交付内容**均为一到两句话。

## 3 Transformer 语言模型架构

语言模型接收批量整数 token ID 序列，即形状 `(batch_size, sequence_length)` 的张量；返回词表上的批量归一概率分布，形状为 `(batch_size, sequence_length, vocab_size)`。序列中每个位置的分布用于预测其下一个词。

训练时，用这些预测计算真实下一词与预测下一词之间的交叉熵。生成时，取最后一个时间步的下一词分布，通过取最大概率或采样等方式生成 token，把它追加到输入序列后重复。

### 3.1 Transformer LM

给定 token ID 序列，模型先通过输入嵌入把 ID 转换为稠密向量，再经过 `num_layers` 个 Transformer 块，最后应用可学习线性投影（输出嵌入或 LM head），产生下一 token 的 logits。整体流程为：

```text
输入 -> Token Embedding -> Transformer Block × num_layers
     -> Norm -> Linear（输出嵌入）-> Softmax -> 输出概率
```

Token 嵌入层接收 `(batch_size, sequence_length)` 的整数张量，输出 `(batch_size, sequence_length, d_model)`。每个 Transformer 块接收并返回 `(batch_size, sequence_length, d_model)`，通过自注意力聚合序列信息，并用前馈层做非线性变换。全部块之后进行最终归一化，再用可学习线性层产生 logits。

### 3.2 说明：批处理、Einsum 与高效计算

Transformer 会在批次元素、序列位置和注意力头等“类批次”维度上重复同类计算。PyTorch 的许多操作能够接收张量开头的额外批次维度并高效广播。例如，形状 `(batch_size, sequence_length, d_model)` 的数据 $D$ 与 `(d_model, d_model)` 的矩阵 $A$ 相乘时，`D @ A` 会把前两个维度当作批次维度。

使用多步 `view`、`reshape`、`transpose` 整理这些形状常常难读。更易用的方案是 `torch.einsum`，或与框架无关的 `einops`、`einx`。`einsum` 可完成任意维度的张量缩并，`rearrange` 可重新排序、拼接或拆分任意维度。课程强烈建议学习 einsum；新手可先用 `einops`，熟悉后可学习更通用的 `einx`。[^4]

原文三个示例分别展示：用

```python
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

代替不易辨认形状的批量矩阵乘法；通过 `rearrange`/`einsum` 广播图像暗化因子；以及把 `(height, width)` 合并为 `pixel` 维度，在各通道独立执行像素混合。其核心优势是：einsum 表达式本身记录了输入和输出轴的含义，具有自文档性，并能自然处理任意前导批次维度。其余张量还可用 `jaxtyping` 等工具添加 `Tensor` 类型提示。性能影响将在作业 2 中讨论。

#### 3.2.1 数学记法与内存顺序

机器学习论文常用行向量，与 NumPy、PyTorch 默认行主序配合：

$$y=xW^\top,\tag{1}$$

其中 $W\in\mathbb{R}^{d_{out}\times d_{in}}$，$x\in\mathbb{R}^{1\times d_{in}}$，批次可放在最外维，形成 $X\in\mathbb{R}^{batch\times d_{in}}$。

线性代数更常用列向量：

$$y=Wx,\tag{2}$$

此时若批处理，批次维度位于最后，形成 $\widetilde X\in\mathbb{R}^{d_{in}\times batch}$。本作业的数学公式主要使用列向量；在 PyTorch 中采用普通矩阵乘法时，要按公式 (1) 转置权重。使用 einsum 时只需正确标注轴。Matlab、Julia、Fortran 使用列主序，而 Python 生态通常使用 C 风格行主序。

### 3.3 基本构件：线性与嵌入模块

#### 3.3.1 参数初始化

不良初始化可能导致梯度消失或爆炸。预归一化 Transformer 虽较稳健，初始化仍影响训练速度和收敛。本作业使用：

- 线性权重：$\mathcal N(0,\frac{2}{d_{in}+d_{out}})$，截断至 $[-3\sigma,3\sigma]$；
- 嵌入：$\mathcal N(0,1)$，截断至 $[-3,3]$；
- RMSNorm：全 1。

用 `torch.nn.init.trunc_normal_` 初始化截断正态权重。

#### 问题（linear）：实现线性模块（1 分）

实现继承 `torch.nn.Module` 且执行

$$y=Wx\tag{3}$$

的 `Linear` 类，不含偏置。接口应类似内置 `nn.Linear`：

- `__init__(self, in_features, out_features, device=None, dtype=None)`；
- `forward(self, x: torch.Tensor) -> torch.Tensor`。

必须调用父类构造函数，以 $W$ 而非 $W^\top$ 的形式把权重存为 `nn.Parameter`，不得使用 `nn.Linear` 或 `nn.functional.linear`。按前述方案初始化。实现 `adapters.run_linear`，可用 `Module.load_state_dict` 载入测试权重，然后运行 `uv run pytest -k test_linear`。

#### 问题（embedding）：实现嵌入模块（1 分）

实现继承 `torch.nn.Module` 的 `Embedding`，不得使用 `nn.Embedding`。它用形状 `(batch_size, sequence_length)` 的 `torch.LongTensor` 索引 `(vocab_size, d_model)` 的嵌入矩阵。接口：

- `__init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`；
- `forward(self, token_ids: torch.Tensor) -> torch.Tensor`。

必须调用父类构造函数，把嵌入矩阵存为 `nn.Parameter`，并令 `d_model` 为最后一维；不得使用 `nn.Embedding` 或 `nn.functional.embedding`。实现 `adapters.run_embedding`，运行 `uv run pytest -k test_embedding`。

### 3.4 预归一化 Transformer 块

每个块有多头自注意力和逐位置前馈网络两个子层。原始 Transformer 在各子层残差连接之后归一化，称为“后归一化”。现代模型通常把归一化移到子层输入，并在最后一个块后额外归一化，形成“预归一化”。这样从输入嵌入到最终输出存在一条未经归一化的干净“残差流”，被认为有助于梯度传播；GPT-3、LLaMA、PaLM 等都采用这种结构。

#### 3.4.1 均方根层归一化

本作业用 RMSNorm。给定 $a\in\mathbb{R}^{d_{model}}$：

$$\operatorname{RMSNorm}(a_i)=\frac{a_i}{\operatorname{RMS}(a)}g_i,\tag{4}$$

$$\operatorname{RMS}(a)=\sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}a_i^2+\varepsilon}.$$

$g_i$ 是 `d_model` 个可学习增益，$\varepsilon$ 通常为 `1e-5`。求平方前把输入提升到 `torch.float32` 以防溢出，结束后转回原数据类型。

#### 问题（rmsnorm）：RMSNorm（1 分）

将 RMSNorm 实现为 `torch.nn.Module`：

- `__init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`；
- `forward(self, x: torch.Tensor) -> torch.Tensor`，输入输出均为 `(batch_size, sequence_length, d_model)`。

实现 `adapters.run_rmsnorm`，运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

原始 Transformer 用两次线性变换，中间为 ReLU，内层通常是输入维度 4 倍。现代 LLM 常用带门控的 SwiGLU，并省略线性层偏置。

$$\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}},\tag{5}$$

$$\operatorname{GLU}(x,W_1,W_2)=\sigma(W_1x)\odot W_2x,\tag{6}$$

$$\operatorname{FFN}(x)=W_2\bigl(\operatorname{SiLU}(W_1x)\odot W_3x\bigr).\tag{7}$$

其中 $W_1,W_3\in\mathbb{R}^{d_{ff}\times d_{model}}$、$W_2\in\mathbb{R}^{d_{model}\times d_{ff}}$，通常 $d_{ff}=\frac83d_{model}$，实际实现可取附近的 64 倍数以提高硬件效率。

#### 问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）

实现由 SiLU 和 GLU 组成的 SwiGLU；本题允许使用 `torch.sigmoid`。令 $d_{ff}$ 约为 $\frac83d_{model}$ 且为 64 的倍数。实现 `adapters.run_swiglu`，运行 `uv run pytest -k test_swiglu`。

#### 3.4.3 相对位置嵌入

实现旋转位置嵌入 RoPE。对位置 $i$ 的查询 $q^{(i)}=W_qx^{(i)}$，计算 $q'^{(i)}=R^iq^{(i)}$。$R^i$ 将成对元素作为二维向量旋转，角度

$$\theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d}},$$

对应 $2\times2$ 块：

$$R_k^i=\begin{pmatrix}\cos\theta_{i,k}&\sin\theta_{i,k}\\-\sin\theta_{i,k}&\cos\theta_{i,k}\end{pmatrix}.\tag{8}$$

不应构造完整 $d\times d$ 矩阵。正弦、余弦可跨层和批次复用，可用 `register_buffer(persistent=False)` 预计算；它们不是可学习参数。键向量按其位置执行相同旋转。

#### 问题（rope）：实现 RoPE（2 分）

实现 `RotaryPositionalEmbedding`：

- `__init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`；
- `forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`。

输入 $x$ 形状 `(..., seq_len, d_k)`，可有任意批次维；`token_positions` 形状 `(..., seq_len)`。按位置切片预计算的正弦、余弦张量。完成 `adapters.run_rope`，运行 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

Softmax 定义为：

$$\operatorname{softmax}(v)_i=\frac{e^{v_i}}{\sum_j e^{v_j}}.\tag{10}$$

为避免大值导致 `inf/inf = NaN`，利用 softmax 对统一平移不变的性质，在指数运算前减去相应维度最大值。

#### 问题（softmax）：实现 softmax（1 分）

编写函数，接收张量与维度 $i$，沿该维应用 softmax，输出形状不变且该维归一化；必须先减去该维最大值。实现 `adapters.run_softmax`，运行 `uv run pytest -k test_softmax_matches_pytorch`。

注意力定义为：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.\tag{11}$$

$Q\in\mathbb{R}^{n\times d_k}$、$K\in\mathbb{R}^{m\times d_k}$、$V\in\mathbb{R}^{m\times d_v}$ 是输入，不是可学习参数。

布尔掩码 $M\in\{\mathrm{True},\mathrm{False}\}^{n\times m}$ 的 `True` 表示查询 $i$ 可以关注键 $j$，`False` 表示不可关注。可在 softmax 前对 `False` 位置加 $-\infty$。

#### 问题（scaled_dot_product_attention）：缩放点积注意力（5 分）

实现应接受形状 `(batch_size, ..., seq_len, d_k)` 的键和查询、`(batch_size, ..., seq_len, d_v)` 的值，并返回 `(batch_size, ..., seq_len, d_v)`。支持可选 `(seq_len, seq_len)` 布尔掩码：允许位置的概率合计为 1，禁止位置概率为 0。实现 `adapters.run_scaled_dot_product_attention`，运行：

```shell
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
```

#### 3.4.5 因果多头自注意力

$$\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h),\tag{12}$$

$$head_i=\operatorname{Attention}(Q_i,K_i,V_i),\tag{13}$$

$$\operatorname{MultiHeadSelfAttention}(x)=W_O\operatorname{MultiHead}(W_Qx,W_Kx,W_Vx).\tag{14}$$

可学习参数为 $W_Q,W_K\in\mathbb{R}^{hd_k\times d_{model}}$、$W_V\in\mathbb{R}^{hd_v\times d_{model}}$、$W_O\in\mathbb{R}^{d_{model}\times hd_v}$。键、值、查询投影应总共使用三次矩阵乘法；拓展目标是合成一次。

因果掩码必须阻止位置 $i$ 关注未来位置 $j>i$，否则训练时会泄漏真实下一词。可用 `torch.triu` 或广播索引比较构造掩码，并复用缩放点积注意力的掩码支持。RoPE 只应用于查询和键，不应用于值；注意力头维度视为批次维，各头使用相同位置旋转。

#### 问题（multihead_self_attention）：因果多头自注意力（5 分）

将其实现为 `torch.nn.Module`，至少接收 `d_model: int` 与 `num_heads: int`，并设 $d_k=d_v=d_{model}/h$。实现 `adapters.run_multihead_self_attention`，运行 `uv run pytest -k test_multihead_self_attention`。

### 3.5 完整 Transformer LM

每个块的两个子层都依次执行 RMSNorm、主要操作（MHA 或 FF）和残差相加。第一个子层为：

$$y=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)).\tag{15}$$

#### 问题（transformer_block）：Transformer 块（3 分）

实现预归一化块，至少接收 `d_model`、`num_heads`、`d_ff`。实现 `adapters.run_transformer_block`，运行 `uv run pytest -k test_transformer_block`。

**交付内容：** 通过测试的 Transformer 块代码。

#### 问题（transformer_lm）：Transformer LM（3 分）

把嵌入、`num_layers` 个块、最终归一化和 LM head 组装起来。除块参数外，还接收：

- `vocab_size: int`：确定 token 嵌入矩阵维度；
- `context_length: int`：确定 RoPE 正弦、余弦缓冲区维度；
- `num_layers: int`：Transformer 块数。

实现 `adapters.run_transformer_lm`，运行 `uv run pytest -k test_transformer_lm`。

**交付内容：** 通过测试的 Transformer LM 模块。

#### 资源核算

Transformer 中绝大多数 FLOPs 来自矩阵乘法。先列出一次前向传播中的全部矩阵乘法，再换算成本。若 $A\in\mathbb{R}^{m\times n}$、$B\in\mathbb{R}^{n\times p}$，则 $AB$ 需要 $2mnp$ FLOPs：每个点积有 $n$ 次乘法和 $n$ 次加法，输出共有 $mp$ 项。

#### 问题（transformer_accounting）：资源核算（5 分）

考虑与 GPT-2 XL 同规模、采用本作业架构的配置：

```text
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288
```

**(a)** 模型有多少可训练参数？单精度加载模型需要多少内存？交付一到两句话。

**(b)** 列出上下文长度为 1,024 时一次前向传播的全部矩阵乘法及说明，并给出总 FLOPs。

**(c)** 根据核算，模型哪些部分消耗最多 FLOPs？

**交付内容：** 一到两句话回答。

**(d)** 对 GPT-2 small（12 层、`d_model=768`、12 头）、medium（24 层、1024、16 头）和 large（36 层、1280、20 头）重复分析。模型增大时，Transformer LM 各部分占总 FLOPs 的比例如何变化？

**交付内容：** 对每种模型给出各组件 FLOPs 及其占前向传播总 FLOPs 的比例；再用一到两句话描述模型规模变化对各组件比例的影响。

**(e)** 把 GPT-2 XL 上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？各组件的相对贡献如何变化？

**交付内容：** 一到两句话回答。

## 4 训练 Transformer LM

现在已有数据预处理（分词器）和模型（Transformer）。还需构建训练支持代码，包括：

- **损失：** 交叉熵；
- **优化器：** 用 AdamW 最小化损失；
- **训练循环：** 加载数据、保存检查点并管理训练。

### 4.1 交叉熵损失

对长度为 $m+1$ 的序列 $x$，Transformer 定义每个 $i=1,\ldots,m$ 的条件分布 $p_\theta(x_{i+1}\mid x_{1:i})$。对训练集 $D$，标准交叉熵（负对数似然）为：

$$
\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i}).\tag{16}
$$

一次 Transformer 前向传播会同时得到所有位置的条件分布。模型在位置 $i$ 产生 logits $o_i\in\mathbb{R}^{vocab\_size}$：

$$
p(x_{i+1}\mid x_{1:i})=\operatorname{softmax}(o_i)[x_{i+1}]
=\frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size}\exp(o_i[a])}.\tag{17}
$$

#### 问题（cross_entropy）：实现交叉熵（1 分）

编写函数，接收预测 logits $o_i$ 与目标 $x_{i+1}$，计算 $\ell_i=-\log\operatorname{softmax}(o_i)[x_{i+1}]$。要求：

- 为保证数值稳定性减去最大元素；
- 尽可能抵消 `log` 与 `exp`；
- 处理任意额外批次维，并返回批次平均值；类批次维始终位于词表维之前。

实现 `adapters.run_cross_entropy`，运行 `uv run pytest -k test_cross_entropy`。

**困惑度。** 若长度为 $m$ 的序列损失为 $\ell_1,\ldots,\ell_m$：

$$
\operatorname{perplexity}=\exp\left(\frac1m\sum_{i=1}^{m}\ell_i\right).\tag{18}
$$

### 4.2 SGD 优化器

最简单的基于梯度的优化器是随机梯度下降。随机初始化参数 $\theta_0$ 后，每一步执行：

$$
\theta_{t+1}\leftarrow\theta_t-\alpha_t\nabla L(\theta_t;B_t),\tag{19}
$$

其中 $B_t$ 是从数据集随机抽取的批次，学习率 $\alpha_t$ 与批量大小 $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

自定义优化器继承 `torch.optim.Optimizer`，必须实现：

- `__init__(self, params, ...)`：初始化优化器，把参数与默认超参数字典传给父类构造函数；`params` 可以是参数集合或采用不同超参数的参数组。
- `step(self)`：在反向传播后原地更新参数。遍历每个参数张量 `p`，根据存在的 `p.grad` 修改 `p.data`。

原文示例实现了学习率随时间衰减的 SGD：

$$
\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{t+1}}\nabla L(\theta_t;B_t).\tag{20}
$$

它在 `self.state[p]` 中保存各参数的迭代计数，按参数组读取 `lr`，跳过没有梯度的参数，并接受可选 `closure` 以符合 PyTorch API。典型训练循环依次执行 `zero_grad()`、前向计算损失、`loss.backward()` 和 `opt.step()`。

#### 问题（learning_rate_tuning）：调整学习率（1 分）

把原文 SGD 玩具示例的学习率改为 `1e1`、`1e2`、`1e3`，各运行 10 步。观察损失是更快下降、更慢下降还是发散。

**交付内容：** 一到两句话描述观察结果。

### 4.3 AdamW

现代语言模型通常使用 Adam 的变体。本作业实现广泛使用的 AdamW：它把权重衰减与梯度更新解耦，在每次迭代中把参数拉向 0。AdamW 为每个参数维护一阶矩与二阶矩的运行估计，以额外内存换取稳定性和收敛性。

除学习率 $\alpha$ 外，超参数 $(\beta_1,\beta_2)$ 控制矩估计，$\lambda$ 控制权重衰减。常见设置是 $(0.9,0.999)$，LLaMA、GPT-3 等大型模型常用 $(0.9,0.95)$。$\varepsilon$（如 $10^{-8}$）用于数值稳定性。

**算法：AdamW 优化器**

1. 初始化可学习参数 $\theta$，令一阶矩 $m=0$、二阶矩 $v=0$；
2. 对 $t=1,\ldots,T$：
   1. 抽取批次 $B_t$；
   2. 计算 $g=\nabla_\theta\ell(\theta;B_t)$；
   3. 计算校正学习率 $\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$；
   4. 权重衰减：$\theta\leftarrow\theta-\alpha\lambda\theta$；
   5. 更新 $m\leftarrow\beta_1m+(1-\beta_1)g$；
   6. 更新 $v\leftarrow\beta_2v+(1-\beta_2)g^2$；
   7. 更新 $\theta\leftarrow\theta-\alpha_t\frac{m}{\sqrt v+\varepsilon}$。

注意 $t$ 从 1 开始。

#### 问题（adamw）：实现 AdamW（2 分）

将 AdamW 实现为 `torch.optim.Optimizer` 子类。`__init__` 接收学习率 $\alpha$、$\beta$、$\varepsilon$、$\lambda$。用基类提供的 `self.state` 字典按 `nn.Parameter` 保存矩估计。实现 `adapters.get_adamw_cls`，运行 `uv run pytest -k test_adamw`。

#### 问题（adamw_accounting）：AdamW 训练资源核算（2 分）

假设全部张量使用 `float32`。

**(a)** AdamW 的峰值内存是多少？分别给出参数、激活、梯度和优化器状态，以 `batch_size` 及模型超参数 `vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads` 表示，并假设 $d_{ff}=\frac83d_{model}$。

激活只需考虑：各 Transformer 块的 RMSNorm；多头注意力中的 QKV 投影、$QK^\top$、softmax、值加权和、输出投影；SwiGLU 中的 $W_1,W_2,W_3$、门分支 SiLU 和逐元素乘积；以及最终 RMSNorm、输出嵌入、logits 上的交叉熵。

**交付内容：** 四类内存的代数表达式及总和。

**(b)** 代入 GPT-2 XL 形状，得到仅依赖 `batch_size` 的表达式。在 80 GB 内存中可容纳的最大批量是多少？

**交付内容：** 形如 $a\cdot batch\_size+b$ 的数值表达式和最大批量。

**(c)** AdamW 单步需要多少 FLOPs？

**交付内容：** 代数表达式及简要理由。

**(d)** 模型 FLOPs 利用率（MFU）是实际吞吐量相对硬件理论峰值 FLOPs 吞吐量的比率。NVIDIA H100 对“float32”（实际为 TF32/bfloat19）理论峰值为 495 TFLOP/s。若达到 50% MFU，在单张 H100 上用批量 1024 训练 GPT-2 XL 400K 步需要多久？假设反向传播 FLOPs 是前向的两倍。

**交付内容：** 训练小时数及简要理由。

### 4.4 学习率调度

训练期间最快降低损失的学习率会变化。Transformer 通常先预热到较大学习率，再逐渐衰减。本作业实现 LLaMA 使用的余弦退火调度。调度器接收当前步 $t$ 等参数，返回该步学习率。

参数包括最大、最小学习率 $\alpha_{max},\alpha_{min}$，预热步数 $T_w$，余弦退火结束步 $T_c$：

- 若 $t<T_w$：$\alpha_t=\frac{t}{T_w}\alpha_{max}$；
- 若 $T_w\le t\le T_c$：

$$
\alpha_t=\alpha_{min}+\frac12\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max}-\alpha_{min});
$$

- 若 $t>T_c$：$\alpha_t=\alpha_{min}$。

#### 问题（learning_rate_schedule）：带预热的余弦学习率（1 分）

编写接收 $t,\alpha_{max},\alpha_{min},T_w,T_c$ 并返回 $\alpha_t$ 的函数。实现 `adapters.get_lr_cosine_schedule`，运行 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

某些训练样本会产生很大梯度，使训练不稳定。给定全部参数的梯度 $g$，计算 $\|g\|_2$。若小于最大值 $M$ 则保持不变；否则按 $\frac{M}{\|g\|_2+\varepsilon}$ 缩小，其中 $\varepsilon$（如 $10^{-6}$）保证数值稳定性。结果范数会略小于 $M$。

#### 问题（gradient_clipping）：实现梯度裁剪（1 分）

函数接收参数列表和最大 $\ell_2$ 范数，原地修改各参数梯度，使用 $\varepsilon=10^{-6}$。实现 `adapters.run_gradient_clipping`，运行 `uv run pytest -k test_gradient_clipping`。

## 5 训练循环

### 5.1 数据加载器

分词后的数据是单个 token 序列 $x=(x_1,\ldots,x_n)$。通常把多个原始文档连接起来，并用 `<|endoftext|>` 等分隔符隔开。

数据加载器把它转换成批次流：每个批次含 $B$ 个长度为 $m$ 的序列及对应的长度为 $m$ 的下一 token 目标。例如 $B=1,m=3$ 时，`([x2,x3,x4], [x3,x4,x5])` 是一个可能批次。

这样任何 $1\le i\le n-m$ 都能产生训练序列，采样简单；所有序列等长，无需填充，可提高硬件利用率；也无需把完整数据集载入内存。

#### 问题（data_loading）：实现数据加载（2 分）

函数接收 token ID 整数 NumPy 数组 $x$、`batch_size`、`context_length` 和 PyTorch 设备字符串（如 `'cpu'`、`'cuda:0'`），返回采样输入和相应下一 token 目标。两个张量形状均为 `(batch_size, context_length)`，并放在指定设备。实现 `adapters.run_get_batch`，运行 `uv run pytest -k test_get_batch`。

> **低资源提示：CPU 或 Apple Silicon**  
> CPU 使用设备字符串 `'cpu'`；Apple Silicon 使用 `'mps'`，并确保数据与模型在同一设备。MPS 资料见 PyTorch MPS 文档和 Apple Metal Performance Shaders 文档。

数据集过大时，可使用 `mmap` 把磁盘文件映射到虚拟内存，并在访问时惰性加载。NumPy 提供 `np.memmap`；若数组由 `np.save` 保存，也可用 `np.load(..., mmap_mode='r')`。训练采样时必须以该模式加载，并指定匹配的 `dtype`。建议显式验证映射数据，例如检查值是否超出词表范围。

### 5.2 检查点

训练中需保存模型，以便任务超时或机器故障后恢复，也便于分析中间模型。检查点至少包含模型权重、状态型优化器（如 AdamW）的状态以及停止时的迭代编号。`nn.Module` 与优化器都有 `state_dict()` 和 `load_state_dict()`；`torch.save(obj,dest)` 与 `torch.load(src)` 可序列化和恢复包含张量及普通 Python 对象的结构。

#### 问题（checkpointing）：实现模型检查点（1 分）

实现：

- `save_checkpoint(model, optimizer, iteration, out)`：把模型、优化器和迭代状态写入路径或二进制文件对象 `out`；
- `load_checkpoint(src, model, optimizer)`：从路径或文件对象恢复模型、优化器状态，并返回保存的迭代编号。

参数类型按原文为 `torch.nn.Module`、`torch.optim.Optimizer`、`int`，以及 `str | os.PathLike | BinaryIO`。实现 `adapters.run_save_checkpoint`、`adapters.run_load_checkpoint`，运行 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

#### 问题（training_together）：整合全部组件（4 分）

**交付内容：** 编写脚本，在用户提供的输入上运行训练循环。建议至少支持：

- 配置模型和优化器的各种超参数；
- 使用 `np.memmap` 以内存高效方式加载大型训练集和验证集；
- 把检查点序列化到用户指定路径；
- 定期记录训练与验证性能，例如输出到控制台和/或 Weights & Biases。

## 6 生成文本

语言模型接收长度为 `sequence_length` 的整数序列，输出 `(sequence_length, vocab_size)` 矩阵，其中每个位置的分布预测其下一个 token。最终线性层输出 logits，需要通过公式 (10) 的 softmax 归一化。

### 解码

给模型提供前缀 token（提示词），取模型最后位置的词表分布，再从中采样下一 token。一步解码为：

$$
P(x_{t+1}=i\mid x_{1:t})=\frac{\exp(v_i)}{\sum_j\exp(v_j)},\tag{21}
$$

$$
v=\operatorname{TransformerLM}(x_{1:t})_t\in\mathbb{R}^{vocab\_size}.\tag{22}
$$

反复采样并把生成 token 追加到下一步输入，直到产生 `<|endoftext|>` 或达到用户指定最大 token 数。

### 解码技巧

小模型有时生成质量很低。两个简单技巧可改善结果。

**温度缩放：**

$$
\operatorname{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_j\exp(v_j/\tau)}.\tag{23}
$$

当 $\tau\to0$，最大 logit 主导分布，softmax 趋近于集中在最大项的 one-hot 向量。

**Nucleus/top-p 采样：** 对温度缩放得到的分布 $q$，取最小索引集合 $V(p)$，使 $\sum_{j\in V(p)}q_j\ge p$，只在该集合内重新归一化：

$$
P(x_{t+1}=i\mid q)=
\begin{cases}
q_i/\sum_{j\in V(p)}q_j,&i\in V(p),\\
0,&\text{其他。}
\end{cases}\tag{24}
$$

可把 $q$ 按概率降序排序，依次取最大项直至累计达到 $p$。

#### 问题（decoding）：解码（3 分）

**交付内容：** 实现语言模型解码函数，建议支持：用户提示词补全，生成到 `<|endoftext|>`；最大生成 token 数；采样前的温度缩放；用户指定阈值的 top-p/nucleus 采样。

## 7 实验

现在把全部内容组合起来，在预训练数据集上训练小型语言模型。

### 7.1 实验运行与交付方式

理解 Transformer 各架构组件最好的方式是亲自修改并运行。实验应快速、可复现且有记录。作业会在约 17M 参数的模型与 TinyStories 上进行多次实验，系统地消融组件、改变超参数，并提交实验日志和每次实验的学习曲线。

为提交损失曲线，应定期评估验证损失，同时记录训练步数与墙钟时间。可使用 Weights & Biases 等日志基础设施。

#### 问题（experiment_log）：实验日志（3 分）

为训练与评估代码创建实验跟踪设施，能按梯度步数和墙钟时间记录实验与损失曲线。

**交付内容：** 实验日志基础设施代码，以及本节下列题目的完整实验日志（记录尝试过的全部内容）。

### 7.2 TinyStories

TinyStories 数据简单、训练快。原文给出的示例是一篇关于小男孩 Ben 在商店发现并买下漂亮花瓶、带回家展示给朋友的儿童故事。

#### 7.2.1 超参数调整

起始配置：

- `vocab_size=10000`。典型词表为数万到数十万；尝试改变它并观察词表与模型行为。
- `context_length=256`。TinyStories 可能无需长序列，OpenWebText 则可尝试变化；观察单步运行时间与最终困惑度。
- `d_model=512`。
- `d_ff=1344`，约为 $\frac83d_{model}$ 且为 64 倍数。
- RoPE $\Theta=10000$。
- 4 层、16 头，约 17M 个非嵌入参数。
- 总处理 token 数约 327,680,000，即 `batch_size × total_steps × context_length`。

通过试验为学习率、预热、AdamW 的 $\beta_1,\beta_2,\varepsilon$ 和权重衰减寻找合适默认值。

#### 7.2.2 整合运行

训练 BPE 分词器、编码训练数据，再交给训练循环。若实现正确高效，上述配置在单张 B200 上约需 20-30 分钟。若明显更慢，应检查数据加载、检查点、验证损失代码是否成为瓶颈，并确认实现正确批量化。

#### 7.2.3 调试模型架构的技巧

- 熟悉 IDE 调试器（如 VSCode/Zed）或 `ipdb`；
- 先尝试过拟合单个小批次，正确实现应能迅速把训练损失降至接近 0；
- 在各组件设置断点，检查中间张量形状；
- 监控激活、权重和梯度范数，确认没有爆炸或消失。

#### 问题（learning_rate）：调整学习率（2 B200 小时，3 分）

**(a)** 对学习率做超参数扫描，报告最终损失；若发散则标明。

**交付内容：** 多个学习率的学习曲线，并解释搜索策略；还需提交一个 TinyStories 每 token 验证损失不高于 1.45 的模型。

> **低资源提示**  
> CPU 或 MPS 上把总 token 数降到 40,000,000，目标验证损失可放宽到 2.00。参考实现在 36 GB M4 Max 上使用 `32 × 5000 × 256 = 40,960,000` token，CPU 用时 1 小时 22 分，MPS 36 分钟，5000 步验证损失 1.80。余弦衰减应恰好在最后一步达到最小学习率。MPS 不要启用 TF32（不要调用 `torch.set_float32_matmul_precision('high')`），某些 PyTorch 2.9.0 MPS 内核会静默出错。CPU 可用 `model=torch.compile(model)`；MPS 可用 `model=torch.compile(model, backend='aot_eager')` 优化反向传播，MPS 暂不支持 Inductor。

**(b)** 经验说最佳学习率位于“稳定性边缘”。研究学习率开始发散的位置与最佳学习率的关系。

**交付内容：** 逐渐增大学习率的曲线，至少包含一次发散，并分析其与收敛速度的关系。

#### 问题（batch_size_experiment）：改变批量大小（1 B200 小时，1 分）

从 1 一直改变到 GPU 内存上限，至少尝试若干中间值，包括 64、128；必要时重新优化学习率。

**交付内容：** 不同批量的学习曲线，以及几句话讨论批量大小对训练的影响。

#### 问题（generate）：生成文本（1 分）

用解码器和训练检查点生成文本；可能需要调整温度、top-p 等参数。

**交付内容：** 至少 256 个 token 的文本（或直到首个 `<|endoftext|>`），简评流畅度，并指出至少两个影响质量的因素。

### 7.3 消融与架构修改

#### 消融 1：层归一化

预归一化块为：

$$z=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)),\tag{25}$$
$$y=z+\operatorname{FFN}(\operatorname{RMSNorm}(z)).\tag{26}$$

原始后归一化为：

$$z=\operatorname{RMSNorm}(x+\operatorname{MultiHeadSelfAttention}(x)),\tag{27}$$
$$y=\operatorname{RMSNorm}(z+\operatorname{FFN}(z)).\tag{28}$$

#### 问题（layer_norm_ablation）：移除 RMSNorm（0.5 B200 小时，1 分）

移除全部 RMSNorm 后训练。此前最佳学习率下会怎样？降低学习率能否恢复稳定？

**交付内容：** 移除 RMSNorm 时及其最佳学习率的学习曲线；几句话评论 RMSNorm 的影响。

#### 问题（pre_norm_ablation）：改为后归一化（0.5 B200 小时，1 分）

把预归一化实现改为后归一化并训练。

**交付内容：** 后归一化与预归一化 Transformer 的对比学习曲线。

#### 消融 2：位置嵌入

因果掩码的仅解码器 Transformer 理论上可以在没有显式位置嵌入时推断相对或绝对位置。比较 RoPE 基线与完全不使用位置嵌入的 NoPE。

#### 问题（no_pos_emb）：实现 NoPE（0.5 B200 小时，1 分）

移除全部位置信息并运行。

**交付内容：** RoPE 与 NoPE 性能对比学习曲线。

#### 消融 3：SwiGLU 与 SiLU

比较默认 SwiGLU 与不带 GLU 门控的：

$$\operatorname{FFN}_{SiLU}(x)=W_2\operatorname{SiLU}(W_1x).\tag{29}$$

SwiGLU 用 $d_{ff}\approx\frac83d_{model}$；无门控 SiLU 基线应设 $d_{ff}=4d_{model}$，以近似匹配三矩阵 SwiGLU 的参数量。

#### 问题（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时，1 分）

**交付内容：** 参数量大致匹配时两种前馈网络的学习曲线，以及几句话讨论结果。

> **低资源提示**  
> 后续将转向更大、更嘈杂的 OpenWebText。GPU 资源有限的在线学生可继续在 TinyStories 上测试修改，以验证损失评估性能。

### 7.4 在 OpenWebText 上运行

OpenWebText 是由网络抓取构造的更标准预训练数据，文本比 TinyStories 更真实、复杂、多样。建议浏览训练集了解网络语料内容。切换数据集后可能需要重新调整学习率、批量大小等超参数。

#### 问题（main_experiment）：在 OWT 上实验（2 B200 小时，2 分）

使用与 TinyStories 相同的模型架构和总训练迭代数，在 OpenWebText 上训练。

**交付内容：** OpenWebText 学习曲线；描述与 TinyStories 损失的差异及解释。还需按 TinyStories 相同格式提交 OWT LM 的生成文本，评价流畅度，并解释为何在模型和计算预算相同的情况下质量更差。

### 7.5 自选修改与排行榜

尝试改进 Transformer 架构，并与同学比较超参数和架构。

#### 排行榜规则

- **运行时间：** 单张 B200 最多 45 分钟，可在 SLURM 或 Modal 提交脚本中强制限制。
- **数据：** 只能使用课程提供的 OpenWebText 训练集。
- 除此之外没有限制。

可参考 Llama 3、Qwen 2.5 等开源 LLM 家族，以及 NanoGPT speedrun 仓库 <https://github.com/KellerJordan/modded-nanogpt>。例如可尝试共享输入、输出嵌入权重；若共享，可能需要降低嵌入/LM head 初始化标准差。先在 OpenWebText 小子集或 TinyStories 上测试，再进行完整 45 分钟运行。注意，排行榜上有效的修改未必能泛化到大规模预训练。

#### 问题（leaderboard）：排行榜（10 B200 小时，6 分）

按上述规则训练模型，目标是在 0.75 B200 小时内最小化验证损失。

**交付内容：** 最终验证损失；横轴明确为墙钟时间且小于 45 分钟的学习曲线；所做修改的说明。排行榜提交应至少优于验证损失 5.0 的朴素基线。提交地址：

<https://github.com/stanford-cs336/assignment1-basics-leaderboard>

## 参考文献

以下保留英文原始题名，便于检索：

1. R. Eldan and Y. Li, “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” 2023.
2. A. Gokaslan et al., “OpenWebText corpus.” 2019.
3. R. Sennrich et al., “Neural Machine Translation of Rare Words with Subword Units,” ACL, 2016.
4. C. Wang et al., “Neural Machine Translation with Byte-Level Subwords.” 2019.
5. P. Gage, “A new algorithm for data compression,” 1994.
6. A. Radford et al., “Language Models are Unsupervised Multitask Learners.” 2019.
7. A. Radford et al., “Improving Language Understanding by Generative Pre-Training.” 2018.
8. A. Vaswani et al., “Attention is All you Need,” NeurIPS, 2017.
9. T. Q. Nguyen and J. Salazar, “Transformers without Tears: Improving the Normalization of Self-Attention,” 2019.
10. R. Xiong et al., “On Layer Normalization in the Transformer Architecture,” ICML, 2020.
11. J. L. Ba et al., “Layer Normalization.” 2016.
12. H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” 2023.
13. B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” NeurIPS, 2019.
14. A. Grattafiori et al., “The Llama 3 Herd of Models.” <https://arxiv.org/abs/2407.21783>
15. A. Yang et al., “Qwen2.5 Technical Report.” 2024.
16. A. Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways.” 2022.
17. D. Hendrycks and K. Gimpel, “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units.” 2016.
18. S. Elfwing et al., “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.”
19. Y. N. Dauphin et al., “Language Modeling with Gated Convolutional Networks.”
20. N. Shazeer, “GLU Variants Improve Transformer.” 2020.
21. J. Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding.” 2021.
22. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.
23. I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” ICLR, 2019.
24. T. B. Brown et al., “Language Models are Few-Shot Learners,” NeurIPS, 2020.
25. J. Kaplan et al., “Scaling Laws for Neural Language Models.” 2020.
26. J. Hoffmann et al., “Training Compute-Optimal Large Language Models.” 2022.
27. A. Holtzman et al., “The Curious Case of Neural Text Degeneration,” ICLR, 2020.
28. Y.-H. H. Tsai et al., “Transformer Dissection: An Unified Understanding for Transformer’s Attention via the Lens of Kernel,” EMNLP-IJCNLP, 2019.
29. A. Kazemnejad et al., “The Impact of Positional Encoding on Length Generalization in Transformers,” NeurIPS, 2023.

[^4]: `einops` 更成熟；若 `einx` 存在限制或错误，可退回 `einops` 配合普通 PyTorch。
