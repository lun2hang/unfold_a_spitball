import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
import random
is_train_set = True
file_name = '../data/train_data_cls.txt' if is_train_set else '../data/eval_data_cls.txt'
with open(file_name, 'rt') as f:
    reader = f.readlines()
    rows = list(reader)
for _ in range(30):
    print(rows[random.randint(1, len(rows)-1)])

# 国家 == 16进制原文首数字字符
# 名字 == MD5签名 
# 名字预测国家 == 签名预测原文首数字字符

class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
    	# 指定训练集和测试集
        file_name = '../data/train_data_cls.txt' if is_train_set else '../data/eval_data_cls.txt'
        with open(file_name, 'rt') as f:
            reader = f.readlines()
            rows = list(reader)
        # 人名
        self.names = [row[:-1].split()[1] for row in rows]
        # 人名序列的长度
        self.length = len(self.names)
        # 人名所对应的国家
        self.countries = [row[:-1].split()[0] for row in rows]
        # 所有国家的集合
        self.country_list = list(sorted(set(self.countries)))
        # 国家-》index生成的字典
        self.country_dict = self.getCountryDict()
        # 国家的数量
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
    	# 返回人名和所对应的国家名
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
    	# 返回人名的长度
        return self.length

    def getCountryDict(self):
        country_dict = {}
        # 遍历数据建立国家的字典
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
    	# 将index转换成国家名
        return self.country_list[index]

    def getCountryNum(self):
    	# 获取不同国家的数量
        return self.country_num

# 隐藏层的维度
HIDDEN_SIZE = 128
BATCH_SIZE = 256
# 0~f de aicii 码是48到102,120个词典大小就够了
TOKEN_DIM = 128
TOKEN_DICT_SIZE = 120
# RNN的层数
N_LAYERS = 2
# 训练的轮数
N_EPOCHS = 100
# 字符长度，也就是输入的维度
N_CHARS = 32
# 是否使用GPU
USE_GPU = False

# 建立训练集的dataloader
train_set = NameDataset(is_train_set=True)
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# 建立测试集的dataloader
test_set = NameDataset(is_train_set=False)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
# 获取国家数
N_COUNTRY = train_set.getCountryNum()



class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        # RNN隐藏层的维度
        self.hidden_size = hidden_size
        # 有多少层RNN
        self.n_layers = n_layers
        # 是否使用双向RNN
        self.n_directions = 2 if bidirectional else 1
        # 建立emb 查找表，字典大小 16.代表每一位16进制，每个token是用 hidden size维度表达
        self.embedding = torch.nn.Embedding(TOKEN_DICT_SIZE, hidden_size)
        # 这里RNN我们使用GRU，输入维度是embedding层的输出维度hidden_size，输出维度也为hidden_size，
        # 整个GRU的输入维度是(seq_length(input_size), batch_size, hidden_size)
        # hidden 的维度是(n_layers * nDirectional, batch_size, hidden_size)
        # 输出的维度是(seq_length, batch_size, hidden_size*nDirectional(双向concat))
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers,
                                bidirectional=bidirectional)
        # 最后一层线性层，输入为hidden_size * self.n_directions，输出为output_size国家数
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_szie):
    	# hidden 的维度是(n_layers * nDirectional, batch_size, hidden_size)
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_szie, self.hidden_size)
        # 返回hidden的向量
        return create_tensor(hidden)

    def forward(self, input, seq_len):
        # batch * seq -> seq * batch 转置
        input = input.t()
        # 提取input第2个维度batch_size
        batch_size = input.size(1)
		# 初始化hidden的向量,(n_layers*n_Direction, batch_size, hidden_size)
        hidden = self._init_hidden(batch_szie=batch_size)
        # embedding层,(seq_length, batch_size, hiddensize)
        # 将序列进行查询embedding vec操作，维度为(seq_length(input_size), batch_size, hidden_size),input_size为字典的大小
        embedding = self.embedding(input)

        # pack them up，打包序列，将序列中非零元素的向量进行拼接，使得GRU单元可以处理长短不一的序列。
        # 需要将输入序列进行降序排序，并记录每个batch的长度。形成seq_length的数组
        gru_input = pack_padded_sequence(embedding, seq_len)
		# 通过GRU层之后的中间变量hidden，和输出output，不懂的可以看看GRU源码
        output, hidden = self.gru(gru_input, hidden)
        # 如果是双向GRU，那么就将前向和反向hidden向量进行concat
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
        # 最后经过一层全连接层
        fc_output = self.fc(hidden_cat)
        # 返回全连接层之后的结果
        return fc_output

class MLPClassifier(torch.nn.Module):
    def __init__(self, max_len, word_dim, class_num):
        super(MLPClassifier, self).__init__()

        self.embedding = torch.nn.Embedding(TOKEN_DICT_SIZE, word_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(max_len * word_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num), 
        )

    def forward(self, x, len):
        x = self.embedding(x)
        out = self.mlp(x.view(x.size(0), -1))
        return out

# 创建训练所需要的张量方法
def make_tensor(names, countries):
	# 通过下面的方法，将名字字符串转换成序列，返回序列(len(arr(name))*len(names))以及序列的长度序列
    sequences_and_lengths = [name2list(name=name) for name in names]
    # 名字的字符串序列
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    # 名字的序列长度所构成的序列
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    # 把国家的int型转成long Tensor
    countries = countries.long()
    # make tensor of name, batch * seq_len
    # 先将batch_size * seq_length 最长序列 填成0向量的Tensor，后面再装入数值，使得所有序列等长，短的尾部填0
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    # 然后我们在将名字序列以及seq_length序列填充值到该张量中去
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    # sort by length to use pack_padded_seq，按照原文长度降序排列，排序依据是seq_length长度，得到新的seq_length以及索引
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    # 返回序列的tensor,序列长度的tensor以及对应国家的tensor
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

# 判断是否使用GPU的方法
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda:0')
        tensor = tensor.to(device)
    return tensor

# 将所有的名字string转换成ASCII列表
def name2list(name):
	# 将名字转换成ASCII标中对应的数字，并返回序列，以及序列长度
    arr = [ord(c) for c in name]
    return arr, len(arr)



def trainModel():
	# 定义总的损失
    total_loss = 0
    # trainloader 一次送一个batch，更新一次参数
    for i, (names, countries) in enumerate(trainloader, 1):
    	# 通过数据集创建原文,有效长度和标签的tensor，有效长度是为了说明补成方形tensor后里面有多少数据
        inputs, seq_lengths, target = make_tensor(names, countries)
        # 定义模型的输出
        output = classifier(inputs, seq_lengths)
        # 比较输出与真实标签的loss
        loss = criterion(output, target)
        # 反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
		# 更新损失
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{i * len(inputs)}/{len(train_set)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
        # 返回训练过程中的损失
        return total_loss


# 实例化模型classifier
#classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYERS)
classifier = MLPClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY)

# if USE_GPU:
#     device = torch.device('cuda:0')
#     classifier.to(device)
# 定义损失函数criterion，使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# 梯度下降使用的Adam算法
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


def testModel():
	# 预测准确的个数
    correct = 0
    # 测试集的大小
    total = len(test_set)
    print('evaluating trained model...')
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
        	# 定义相关的tensor
            inputs, seq_lengths, target = make_tensor(names, countries)
            # 模型的输出
            output = classifier(inputs, seq_lengths)
            # 取线性层输出最大的一个作为分类的结果
            pred = output.max(dim=1, keepdim=True)[1]
            # 正确预测个数的累加
            correct += pred.eq(target.view_as(pred)).sum().item()
        # 计算准确率
        percent = '%.2f' % (100 * correct / total)
        print(f'test set: accuracy {correct}/{total}\n{percent}%')
    # 返回模型测试集的准确率
    return correct / total


print('training for %d epochs.' % N_EPOCHS)
acc_list = []
for epoch in range(1, N_EPOCHS + 1):
	trainModel()
	acc = testModel()
	acc_list.append(acc)
	print('acc_list: ', acc_list)
    

torch.save(classifier.state_dict(), 'name_classifier_model.pt')


classifier.load_state_dict(torch.load('name_classifier_model.pt'))


def predict_country(name):
	# 同上，名字序列和长度，这里长度为1，因为输入的是单一的名字
    sequences_and_lengths = [name2list(name=name)]
    # 名字的序列映射
    name_sequences = [sequences_and_lengths[0][0]]
    # 序列长度的张量
    seq_lengths = torch.LongTensor([sequences_and_lengths[0][1]])
    print('sequences_and_lengths: ', sequences_and_lengths)
	# 创建序列的张量
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    #名字的张量
    inputs = create_tensor(seq_tensor)
    # seq_length的张量
    seq_lengths = create_tensor(seq_lengths)
    # 通过模型进行预测输出output张量
    output = classifier(inputs, seq_lengths)
    # 通过线性层的输出取最大项作为预测项输出
    pred = output.max(dim=1, keepdim=True)[1]
    # 返回预测的index
    return pred.item()



