import _001_text_splitter
import _002_SimpleTokenizerV1

vocab = _001_text_splitter.vocab
tokenizer = _002_SimpleTokenizerV1.SimpleTokenizerV1(vocab)
text =  """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))   # KeyError: 'Hello'
