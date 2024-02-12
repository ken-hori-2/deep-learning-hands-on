import torch
from pytorch_transformers import BertForMaskedLM
from pytorch_transformers import BertTokenizer

# Section2

# msk_model = BertForMaskedLM.from_pretrained('bert-base-uncased')  # 訓練済みパラメータの読み込み
# print(msk_model)

text = "I have a pen. I have an apple."

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
words = tokenizer.tokenize(text)
print(words)

text = "[CLS] I played baseball with my friends at school yesterday [SEP]"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
words = tokenizer.tokenize(text)
print(words)

msk_idx = 3 # baseballをMASK
words[msk_idx] = "[MASK]"  # 単語を[MASK]に置き換える
print(words)

word_ids = tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
word_tensor = torch.tensor([word_ids])  # テンソルに変換
print(word_tensor)

msk_model = BertForMaskedLM.from_pretrained("bert-base-uncased") # 重い
# msk_model.cuda()  # GPU対応
msk_model.eval()

x = word_tensor # .cuda()  # GPU対応
y = msk_model(x)  # 予測
result = y[0]
print(result.size())  # 結果の形状

_, max_ids = torch.topk(result[0][msk_idx], k=5)  # 最も大きい5つの値
result_words = tokenizer.convert_ids_to_tokens(max_ids.tolist())  # インデックスを単語に変換
print(result_words)