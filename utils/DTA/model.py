import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

embedding_num_drug = 64      #字典序
embedding_num_target = 25    #字典序
embedding_dim_drug = 16      #2^6
embedding_dim_target = 16    #2^6
hyber_para = 16              #2^4
qubits_cirAorB = 4           #每一边的qubits数
dim_embed = hyber_para

class ClassicalPre(nn.Module):
    def __init__(self, embedding_num_drug=embedding_num_drug, embedding_dim_drug=dim_embed):
        super().__init__()
        self.embed_drug = nn.Embedding(embedding_num_drug, embedding_dim_drug, padding_idx=0)
        self.embed_target = nn.Embedding(embedding_num_target, embedding_dim_target, padding_idx=0)
    def datapre(self, data):
        #drug, target, label = data['smiles'], data['sequence'], data['label']
        drug, target= data['smiles'], data['sequence']
        VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}
        VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

        drugint = [VOCAB_LIGAND_ISO[s] for s in drug]
        targetint = [VOCAB_PROTEIN[s] for s in target]

        if len(drugint) < 128:
            drugint = np.pad(drugint, (0, 128 - len(drugint)))
        else:
            drugint = drugint[:128]

        if len(targetint) < 512:
            targetint = np.pad(targetint, (0, 512 - len(targetint)))
        else:
            targetint = targetint[:512]
            
        drugint, targetint= torch.tensor(drugint, dtype=torch.long), torch.tensor(targetint, dtype=torch.long)

        #drugint, targetint, exp = torch.tensor(drugint, dtype=torch.long), torch.tensor(targetint, dtype=torch.long), \
        #                          torch.tensor(label, dtype=torch.float).unsqueeze(-1)

        d = self.embed_drug(drugint)
        t = self.embed_target(targetint)
        Gram_d = d.T @ d
        Gram_t = t.T @ t
        C_input_d = Gram_d.view(-1, hyber_para, hyber_para)
        C_input_t = Gram_t.view(-1, hyber_para, hyber_para)
        return C_input_d, C_input_t    # 都是[1,16,16]

    def forward(self, data):  # data是一整条数据
        drug_input, target_input = self.datapre(data)
        return drug_input, target_input  #都是[1,16,16]

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1 = nn.Linear(128,32)
        self.FC2 = nn.Linear(32,1)
    def forward(self,x):
        out = F.leaky_relu(self.FC1(x))
        out = F.leaky_relu(self.FC2(out))
        return out

class CNNLayerBased(nn.Module):
    def __init__(self, embedding_num_drug, embedding_num_target, embedding_dim_drug=dim_embed,
                  embedding_dim_target=dim_embed, conv1_out_dim = qubits_cirAorB):
        super().__init__()
        self.data_pre = ClassicalPre()
        self.drugconv1d = nn.Conv1d(embedding_dim_drug, conv1_out_dim, kernel_size = 4, stride = 1, padding = 'same')
        self.targetconv1d = nn.Conv1d(embedding_dim_target, conv1_out_dim, kernel_size = 4, stride = 1, padding = 'same')
        self.linearlayer = Linear()
    def forward(self,x):   #x是一条记录
        drug_input, target_input = self.data_pre(x)
        drug_output = self.drugconv1d(drug_input)   #1 4 16

        target_output = self.targetconv1d(target_input)   #1 4 16

        linear_input = torch.cat([drug_output, target_output],dim=0).view(1,-1)
        linear_output = self.linearlayer(linear_input)
        affinity = linear_output.view(1)
        return affinity

modelDTA=CNNLayerBased(64,25)

#modelDTA.load_state_dict(torch.load('./checkpoints/cnn_dta.pth'))
