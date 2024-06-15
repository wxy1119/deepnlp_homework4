import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

# 定义数据集
class CorpusDataset(Dataset):
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx,device):
        self.source_data = source_data
        self.target_data = target_data
        self.source_word_2_idx = source_word_2_idx
        self.target_word_2_idx = target_word_2_idx
        self.device = device

    def __getitem__(self, index):
        src = self.source_data[index]
        tgt = self.target_data[index]

        src_index = [self.source_word_2_idx[i] for i in src]
        tgt_index = [self.target_word_2_idx[i] for i in tgt]

        return src_index, tgt_index

    def batch_data_alignment(self, batch_datas):
        global device
        src_index, tgt_index = [], []
        src_len, tgt_len = [], []

        for src, tgt in batch_datas:
            src_index.append(src)
            tgt_index.append(tgt)
            src_len.append(len(src))
            tgt_len.append(len(tgt))
        src_max_len = max(src_len)
        tgt_max_len = max(tgt_len)

        # 填充至最大长度，包括 <BOS> 和 <EOS>
        src_index = [[self.source_word_2_idx["<BOS>"]] +
                     src +
                     [self.source_word_2_idx["<PAD>"]] * (src_max_len - len(src))
                     for src in src_index]
        # 先处理可能过长的目标序列
        clipped_tgt_index = [[self.target_word_2_idx["<BOS>"]] +
                             tgt[:tgt_max_len-2] +
                             [self.target_word_2_idx["<EOS>"]]
                             for tgt in tgt_index if len(tgt) + 2 > tgt_max_len]

        # 处理正常长度的目标序列
        normal_tgt_index = [[self.target_word_2_idx["<BOS>"]] +
                            tgt +
                            [self.target_word_2_idx["<EOS>"]] +
                            [self.target_word_2_idx["<PAD>"]] * (tgt_max_len - len(tgt) - 2)
                            for tgt in tgt_index if len(tgt) + 2 <= tgt_max_len]

        # 合并处理过的序列
        tgt_index = clipped_tgt_index + normal_tgt_index
        #print(f"Max src length in this batch: {src_max_len}")
        #print(f"Max tgt length in this batch: {tgt_max_len}")

        #for src in src_index:
        #print(f"Src length before padding: {len(src)}, after padding: {src_max_len}")

        #for tgt in tgt_index:
        #print(f"Tgt length before padding: {len(tgt)}, after padding: {tgt_max_len}")

        #print(f"Shape of src_index after padding: {torch.tensor(src_index).shape}")
        #print(f"Shape of tgt_index after padding: {torch.tensor(tgt_index).shape}")
        # 现在可以安全地将填充后的列表转换为张量
        src_index = torch.tensor(src_index, dtype=torch.long, device=device)
        tgt_index = torch.tensor(tgt_index, dtype=torch.long, device=device)

        return src_index, tgt_index

    def __len__(self):
        assert len(self.source_data) == len(self.target_data)
        return len(self.target_data)

# 定义位置编码和Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.dropout_layer = nn.Dropout(p=dropout)  # 正确初始化dropout层
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
                                          , dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(embed_size, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # 在forward方法内
        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.dropout_layer(src)  # 使用正确的dropout实例
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(self.embed_size)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.decoder(output)
        return output

# 确保d_model的值已知，这里假设model对象有属性d_model
d_model = 256

# 生成文本的函数
# 生成文本的函数
def generate_sentence_transformer(sentence, model, max_len=40):
    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)

    with torch.no_grad():
        # 编码源句子
        memory = model.encoder(src_index) * math.sqrt(d_model)  # 确保sqrt(d_model)正确使用
        memory = model.pos_encoder(memory)

        # 初始化目标序列
        outs = [target_word_2_idx["<BOS>"]]
        tgt_tensor = torch.tensor([outs], device=device)  # 初始化

        for i in range(max_len - 1):
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

            # 位置编码并确保维度正确
            tgt_tensor_with_pe = model.pos_encoder(model.encoder(tgt_tensor) * math.sqrt(d_model))

            # 确保tgt_tensor_with_pe的维度与memory匹配
            out = model.transformer(memory, tgt_tensor_with_pe, tgt_mask=tgt_mask)
            next_word_probs = model.decoder(out[:, -1, :])  # 获取最后一个时间步的输出
            next_word_idx = next_word_probs.argmax(dim=-1).item()
            temperature = 0.6  # 调节温度值，较低的值使模型更倾向于高概率词，较高的值增加随机性
            next_word_probs = next_word_probs / temperature
            next_word_probs = next_word_probs.softmax(dim=-1)
            next_word_idx = torch.multinomial(next_word_probs, 1).item()
            # 打印新生成的单词及其概率
            #print("Generated word:", target_idx_2_word[next_word_idx], "Probability:", next_word_probs.max().item())
            if next_word_idx == target_word_2_idx["<EOS>"]:
                break

            outs.append(next_word_idx)
            new_token = torch.tensor([[next_word_idx]], device=device)
            tgt_tensor = torch.cat((tgt_tensor, new_token), dim=1)

    return "".join([target_idx_2_word[i] for i in outs if i != target_word_2_idx["<BOS>"]])

# 检查词汇表构建过程
def check_vocab_construction(word_2_idx_dict, idx_2_word_list):
    print("Word to Index mapping:")
    for word, idx in word_2_idx_dict.items():
        print(f"Word: {word}, Index: {idx}")

    print("\nIndex to Word mapping:")
    for idx, word in enumerate(idx_2_word_list):
        print(f"Index: {idx}, Word: {word}")

# 检查数据集的索引转换
def check_dataset_indexing(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx):
    for i in range(5):  # 检查前五个样本
        src_sentence = source_corpus[i]
        tgt_sentence = target_corpus[i]

        src_index = [source_word_2_idx[word] for word in src_sentence]
        tgt_index = [target_word_2_idx[word] for word in tgt_sentence]

        print(f"Source Sentence: {src_sentence}")
        print(f"Source Indexes: {src_index}")
        print(f"Target Sentence: {tgt_sentence}")
        print(f"Target Indexes: {tgt_index}\n")

# 检查数据加载器
def check_dataloader(dataloader):
    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Batch {batch_idx + 1}:")
            print(f"Source Batch Shape: {src_batch.shape}")
            print(f"Target Batch Shape: {tgt_batch.shape}")
            print(f"Source Batch: {src_batch}")
            print(f"Target Batch: {tgt_batch}\n")
        else:
            break

# 检查特殊标记的应用
def check_special_tokens_application(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx):
    for i in range(5):  # 检查前五个样本
        src_sentence = source_corpus[i]
        tgt_sentence = target_corpus[i]

        src_index = [source_word_2_idx["<BOS>"]] + [source_word_2_idx[word] for word in src_sentence] + [source_word_2_idx["<PAD>"]]
        tgt_index = [target_word_2_idx["<BOS>"]] + [target_word_2_idx[word] for word in tgt_sentence] + [target_word_2_idx["<EOS>"]]

        print(f"Source Sentence: {src_sentence}")
        print(f"Source Indexes with Special Tokens: {src_index}")
        print(f"Target Sentence: {tgt_sentence}")
        print(f"Target Indexes with Special Tokens: {tgt_index}\n")



if __name__ == '__main__':
    ### configs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    num_corpus = 300
    num_test_corpus = 10
    txt_file_path = "./dataset/天龙八部.txt"
    num_epochs = 100
    lr = 0.001

    dim_encoder_embedding = 256  # 嵌入维度，需为8的倍数
    dim_encoder_hidden = 512  # 可以选择与d_model相同或不同，但不受num_heads限制

    ### data preparation
    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"

    source_target_corpus_ori = []

    with open(txt_file_path, "r", encoding="ansi", errors="ignore") as tmp_file:
        tmp_file_context = tmp_file.read()
        for tmp_char in char_to_be_replaced:
            tmp_file_context = tmp_file_context.replace(tmp_char, "")
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")

        tmp_file_sentences = tmp_file_context.split("。")
        for tmp_idx, tmp_sentence in enumerate(tmp_file_sentences):
            if ("她" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):
                #if (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):
                source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))

    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)
    source_corpus, target_corpus = [], []
    for idx in sample_indexes:
        source_corpus.append(source_target_corpus_ori[idx][0])
        target_corpus.append(source_target_corpus_ori[idx][1])

    test_corpus = []
    for idx in range(len(source_target_corpus_ori)):
        if idx not in sample_indexes:
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))
    test_corpus = random.sample(test_corpus, num_test_corpus)
    test_source_corpus, test_target_corpus = [], []
    for tmp_src, tmp_tgt in test_corpus:
        test_source_corpus.append(tmp_src)
        test_target_corpus.append(tmp_tgt)

    ### one-hot dict generation
    idx_cnt = 0
    word_2_idx_dict = dict()
    idx_2_word_list = list()
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:
        for tmp_sentence in tmp_corpus:
            for tmp_word in tmp_sentence:
                if tmp_word not in word_2_idx_dict.keys():
                    word_2_idx_dict[tmp_word] = idx_cnt
                    idx_2_word_list.append(tmp_word)
                    idx_cnt += 1

    one_hot_dict_len = len(word_2_idx_dict)
    word_2_idx_dict.update({"<PAD>": one_hot_dict_len, "<BOS>": one_hot_dict_len + 1, "<EOS>": one_hot_dict_len + 2})
    idx_2_word_list += ["<PAD>", "<BOS>", "<EOS>"]
    one_hot_dict_len += 3

    source_word_2_idx, target_word_2_idx = word_2_idx_dict, word_2_idx_dict
    source_idx_2_word, target_idx_2_word = idx_2_word_list, idx_2_word_list
    source_corpus_len, target_corpus_len = one_hot_dict_len, one_hot_dict_len

    ### Data Preparation Checks
    #print("Step 1: Checking Vocabulary Construction")
    #check_vocab_construction(word_2_idx_dict, idx_2_word_list)

    #print("\nStep 2: Checking Dataset Indexing")
    #check_dataset_indexing(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx)

    #print("\nStep 3: Checking Special Tokens Application")
    #check_special_tokens_application(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx)

    ### dataloader
    dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx, device)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_alignment)

    #print("\nStep 4: Checking DataLoader")
    #check_dataloader(dataloader)

    ### model
    transformer_model = TransformerModel(
        vocab_size=one_hot_dict_len,
        embed_size=dim_encoder_embedding,
        num_heads=8,
        num_layers=2,
        hidden_dim=dim_encoder_hidden,
        dropout=0.1
    ).to(device)

    ### train model
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        for step, (src_index, tgt_index) in enumerate(dataloader):
            src_index = src_index.clone().detach().to(device)
            tgt_index = tgt_index.clone().detach().to(device)

            optimizer.zero_grad()
            output = transformer_model(src_index, tgt_index)
            output = output.permute(1, 0, 2)

            # 计算损失，注意这里需要过滤掉 <PAD> 标记
            loss = nn.CrossEntropyLoss(ignore_index=word_2_idx_dict["<PAD>"], reduction='mean')(output.reshape(-1, one_hot_dict_len), tgt_index.reshape(-1))
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))

    # Visualize training loss
    plt.figure()
    plt.plot(np.arange(1, num_epochs + 1), losses, "b-")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss of Transformer")
    plt.savefig("./training_loss_transformer.png")
    plt.show()

    ### generate sentences from test corpus
    for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
        tmp_generated_sentence = generate_sentence_transformer(tmp_src_sentence, transformer_model)
        print("----------------Result {}----------------".format(idx + 1))
        print("Source sentence: {}".format(tmp_src_sentence))
        print("True target sentence: {}".format(tmp_gt_sentence))
        print("Generated target sentence: {}".format(tmp_generated_sentence))
