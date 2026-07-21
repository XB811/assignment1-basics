### 2、BPE分词器

#### 2.1 Unicode

**(a)** `chr(0)` 返回哪个 Unicode 字符？

```python
>>> chr(0)
'\x00'
```

**(b)** 这个字符的字符串表示（`__repr__()`）与打印出来的表示有何不同？

```bash
>>> chr(0)
'\x00'
>>> ord('\x00')
0
>>> ord('\x00').__repr__()
'0'
>>> repr('\x00')
"'\\x00'"
>>> '\x00'.__repr__()
"'\\x00'"
>>> 
```



**(c)** 当这个字符出现在文本中时会发生什么？可以在 Python 解释器中尝试以下内容，观察结果是否符合预期：

```
>>> chr(0)
'\x00'  # 字符不可见，显示字符的编码
>>> print(chr(0)) # 直接打印字符，但是不可见

>>> "this is a test" + chr(0) + "string" # 同chr(0)
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string") # 同print(chr(0))
this is a teststring
>>> 

```

#### 2.2 Unicode编码

**(a)** 与 UTF-16 或 UTF-32 相比，选择在 UTF-8 编码的字节上训练分词器有哪些理由？比较不同输入字符串在这些编码下的输出可能会有帮助。

- utf-8 兼容ASCII

- utf-8 使用动态的编码，字节数为 1~4个字节，utf-16 为 2或4个字节，utf-32为4个字节。 理论上utf-8 编码使用的字节数最少

(b) 考虑下面这个错误的函数。它原本想把 UTF-8 字节串解码成 Unicode 字符串。该函数为什么不正确？请给出一个会产生错误结果的输入字节串示例。

```
>>> decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
>>> 
```

utf-8中只有**ASCII** 是单个字节的，其他的字符编码都是多字节的，`decode_utf8_bytes_to_str_wrong`是逐个字节转义的，不能转义非**ASCII**编码字符

（c） 给出一个无法解码为任何 Unicode 字符的双字节序列。

```
>>> print(b'\x01\x01'.decode("utf-8"))

>>> print(b'\xd1\xb1'.decode("utf-8"))
ѱ
>>> 
```



utf-8编码规则如下，不符合改编码规则的不能解释为Unicode 字符

1 字节：`0xxxxxxx`

2 字节：`110xxxxx 10xxxxxx`

3 字节：`1110xxxx 10xxxxxx 10xxxxxx`

4 字节：`11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`

### 2.4 BPE 分词器训练

预分词（pre-tokenization）是在正式进行 BPE 合并前，先把语料切成较粗粒度的片段。它主要有三个意义：

1. **限制 BPE 合并范围**

   BPE 只在每个预 token 内合并，不跨边界合并。例如空格、标点或文档边界可以阻止两个不相关片段被合成一个 token。

2. **提高训练效率**

   基于预分词得到的频率表进行合并词对统计，不需要遍历整个原始字符串，就能统计出合并词对的出现频率

   可以先统计每种预 token 出现多少次，再按出现次数计算其中的字节对频率。某个词出现 100 次时，不必扫描处理 100 遍，而可以处理一次后乘以频数。

3. **让词表更合理**

   如果直接在整个字节流上合并，可能产生大量跨单词或混合标点的 token。预分词使合并结果更倾向于有用的词、词根和子词单元。

需要注意：预 token 只是 BPE 训练和编码时的临时边界，不一定就是最终 token。一个预 token 之后仍可能被 BPE 拆成多个 token，也可能被完全合并为一个 token。



##### 2.5

 uv run pytest tests/test_train_bpe.py 已通过测试点

问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）

Amd9700x + 40G memory / 16 线程：time: 0:01:05.132131

性能分析运行代码： uv run .\cs336_basics\section2\train_bpe.py --profile   

###### ai 分析结果：

**优化1：**str 转tuple(bytes,...)

```python
BYTE_SYMBOLS: tuple[bytes, ...] = tuple(bytes((i,)) for i in range(256))
tuple(BYTE_SYMBOLS[x] for x in token.encode("utf-8"))
# 预加载 0~255的数字到 bytes的映射，减少访问次数
```



**优化2：**先预分词成功后，再转为tuple(bytes,...)

```python
        pre_token_counts: Counter[str] = Counter()

        special_pattern = re.compile(pattern)
        pre_token_pattern = re.compile(PAT)

        for one_chunk in special_pattern.split(chunk):
            pre_token_counts.update(
                match.group()
                for match in pre_token_pattern.finditer(one_chunk)
            )

        frequency_table = Counter({
            tuple(BYTE_SYMBOLS[x] for x in token.encode("utf-8")): count
            for token, count in pre_token_counts.items()
        })
```

**优化3：**预编译正则

```
special_pattern = re.compile(pattern)
pre_token_pattern = re.compile(PAT)
```

