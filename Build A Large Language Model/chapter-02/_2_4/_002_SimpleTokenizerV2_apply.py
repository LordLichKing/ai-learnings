import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from _2_3 import _001_text_splitter
import _001_SimpleTokenizerV2

all_tokens = sorted(list(set(_001_text_splitter.preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer = _001_SimpleTokenizerV2.SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

"""
一般地，在不同大语言模型中，研究人员可能会考虑引入如下特殊词元：
1) [BOS] (序列开始): 标记文本起点，告知大语言模型一段内容的开始
2) [EOS] (序列结束): 位于文本的末尾，类似<|endoftext|>，特别适用于连接多个不相关的文本。
3) [PAD] (填充): 当使用批次大小(batch size)大于1的批量数据训练大语言模型时，数据中的文本长度可能不同。
   为了使所有文本具有相同的长度，较短的文本会通过添加[PAD]词元进行拓展或“填充”，以匹配批量数据中的最长文本长度。
"""