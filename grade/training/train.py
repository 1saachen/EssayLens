# ================================================================
# FusionGraphBERT-AES: 结合图神经网络与特征工程的自动作文评分系统
# ================================================================

import os
import hashlib
import json
import random
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from textstat.textstat import textstat

# ---------------------------
# -------- CONFIG -----------
# ---------------------------
DATASET_PATH = "./ASAP2_train_sourcetexts.csv"
GRAPH_CACHE_DIR = "graphs_cache"
FEATURE_CACHE_DIR = "features_cache"
PROMPT_CACHE_DIR = "prompt_cache"
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)

# 模型配置
MODEL_NAME = "microsoft/deberta-v3-base"
# 修改tokenizer加载方式
try:
    # 首先尝试使用慢速tokenizer
    from transformers import DebertaV2Tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    print("✓ DebertaV2Tokenizer (slow) loaded successfully")
except Exception as e:
    print(f"DebertaV2Tokenizer failed: {e}")
    try:
        # 如果失败，尝试使用AutoTokenizer但禁用fast版本
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print("✓ AutoTokenizer with use_fast=False loaded")
    except Exception as e2:
        print(f"All tokenizer attempts failed: {e2}")
        # 最后回退到其他模型
        print("Falling back to roberta-base...")
        MODEL_NAME = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 20
LR_BERT = 5e-6
LR_GRAPH = 1e-4
LR_FEATURES = 1e-3
DROPOUT = 0.3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCUM_STEPS = 2
FREEZE_BERT_LAYERS = 6

# 图编码器配置
GRAPH_IN_DIM = 2
GRAPH_HIDDEN = 64
GRAPH_OUT = 256

# 特征工程配置
NUM_FEATURES = 20

# Prompt相关度配置
PROMPT_RELEVANCE_WEIGHT = 0.15

print("Using device:", DEVICE)

# ---------------------------
# ---- 特征工程函数 ----
# ---------------------------
class FeatureExtractor:
    def __init__(self):
        self.dale_chall_common_words = self._load_dale_chall_words()
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    
    def _load_dale_chall_words(self):
        """加载常用词表"""
        dale_chall_url = "https://gist.githubusercontent.com/Abhishek-P/e00edcc6f508640fe24f263f5836a7dc/raw/166225e09fb8b554deff37ec344ad5ca40dab2fb/dale-chall-3000-words.txt"
        try:
            import requests
            response = requests.get(dale_chall_url, timeout=10)
            if response.status_code == 200:
                return set(response.text.splitlines())
        except:
            pass
        return set()
    
    def safe_word_tokenize(self, text):
        """安全的分词函数"""
        try:
            return nltk.word_tokenize(str(text))
        except:
            try:
                return str(text).split()
            except:
                return []
    
    def extract_features(self, text):
        """提取20维语言学特征"""
        if pd.isna(text) or text == "" or text is None:
            return np.zeros(NUM_FEATURES, dtype=float)
        
        try:
            text_str = str(text)
            tokens = self.safe_word_tokenize(text_str)
            words = [t for t in tokens if re.match(r"\w+", t)]
            n_chars = len(text_str)
            n_words = len(words)
            n_sent = max(1, text_str.count('.') + text_str.count('!') + text_str.count('?'))
            uniq = len(set(w.lower() for w in words))
            
            # 词性标注
            try:
                pos = nltk.pos_tag(words)
                pos_counts = Counter(tag for _, tag in pos)
            except:
                pos_counts = Counter()

            # 基础特征
            ch = n_chars
            w = n_words
            co = text_str.count(',')
            uw = uniq

            # 词性特征
            nnp = pos_counts.get('NNP', 0)
            dt = pos_counts.get('DT', 0)
            nn = pos_counts.get('NN', 0)
            rb = pos_counts.get('RB', 0)
            jj = pos_counts.get('JJ', 0)
            inn = pos_counts.get('IN', 0)

            # 可读性特征
            try:
                fog = textstat.gunning_fog(text_str) if n_words > 0 else 0
            except:
                fog = 0
            try:
                smog = textstat.smog_index(text_str) if n_words > 0 else 0
            except:
                smog = 0
            try:
                rix = textstat.rix(text_str) if n_words > 0 else 0
            except:
                rix = 0
            try:
                dc = textstat.dale_chall_readability_score(text_str) if n_words > 0 else 0
            except:
                dc = 0
                
            wt = len(set(words))
            s = n_sent
            lw = sum(1 for w in words if len(w) > 6)
            
            try:
                cw = sum(1 for w in words if textstat.syllable_count(w) > 2)
            except:
                cw = 0
                
            nbw = sum(1 for w in words if w.lower() not in self.dale_chall_common_words)
            
            try:
                dw = sum(1 for w in words if len(textstat.difficult_words_list([w])) > 0)
            except:
                dw = 0

            feats = [ch, w, co, uw, nnp, dt, nn, rb, jj, inn, fog, smog, rix, dc, wt, s, lw, cw, nbw, dw]
            return np.array(feats, dtype=float)
            
        except Exception as e:
            print(f"Error computing features: {e}")
            return np.zeros(NUM_FEATURES, dtype=float)

feature_extractor = FeatureExtractor()

# ---------------------------
# ---- 图构建和缓存 ----
# ---------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_graph_from_text(text: str, max_nodes: int = 256) -> Data:
    """构建依赖图"""
    doc = nlp(text)
    nodes = list(doc)[:max_nodes]
    if len(nodes) == 0:
        x = torch.zeros((1, GRAPH_IN_DIM), dtype=torch.float32)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    edges = []
    for token in nodes:
        head_idx = token.head.i
        if token.i != head_idx and head_idx < max_nodes:
            edges.append((token.i, head_idx))
    if len(edges) == 0:
        edges = [(0, 0)]

    pos_ids = [int(tok.pos) for tok in nodes]
    is_root = [1 if tok.dep_ == "ROOT" else 0 for tok in nodes]
    x = torch.tensor(list(zip(pos_ids, is_root)), dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def get_cached_graph(text: str) -> Data:
    h = text_hash(text)
    path = os.path.join(GRAPH_CACHE_DIR, f"{h}.pt")
    if os.path.exists(path):
        try:
            return torch.load(path, weights_only=False)
        except:
            pass
    g = build_graph_from_text(text, max_nodes=256)
    torch.save(g, path)
    return g

def get_cached_features(text: str) -> np.ndarray:
    h = text_hash(text)
    path = os.path.join(FEATURE_CACHE_DIR, f"{h}.npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except:
            pass
    features = feature_extractor.extract_features(text)
    np.save(path, features)
    return features

# ---------------------------
# ---- Prompt相关度模块 ----
# ---------------------------
class FixedPromptRelevanceModule(nn.Module):
    """修复的Prompt相关度评估模块 - 解决输出饱和问题"""
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 使用更简单的架构，避免饱和
        self.essay_pool = nn.AdaptiveAvgPool1d(1)
        self.prompt_pool = nn.AdaptiveAvgPool1d(1)
        
        # 相似度计算 - 使用更小的网络
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 4),
            nn.Tanh(),  # 使用Tanh避免饱和
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # 更保守的初始化
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更小的初始化权重
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, essay_embeddings, prompt_embeddings, essay_mask=None, prompt_mask=None):
        batch_size = essay_embeddings.size(0)
        
        # 使用平均池化而不是加权平均，更稳定
        if essay_mask is not None:
            # 应用mask的池化
            essay_masked = essay_embeddings * essay_mask.unsqueeze(-1)
            essay_pooled = essay_masked.sum(dim=1) / essay_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            essay_pooled = essay_embeddings.mean(dim=1)
            
        if prompt_mask is not None:
            prompt_masked = prompt_embeddings * prompt_mask.unsqueeze(-1)
            prompt_pooled = prompt_masked.sum(dim=1) / prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            prompt_pooled = prompt_embeddings.mean(dim=1)
        
        # 计算余弦相似度作为基础
        cosine_sim = F.cosine_similarity(essay_pooled, prompt_pooled, dim=1).unsqueeze(1)
        
        # 组合特征
        combined = torch.cat([essay_pooled, prompt_pooled], dim=1)
        
        # 计算相关度分数 - 基于组合特征和余弦相似度
        base_relevance = self.similarity_net(combined).squeeze(-1)
        
        # 结合余弦相似度，避免网络饱和
        relevance_score = 0.3 * base_relevance + 0.7 * cosine_sim.squeeze(-1)
        
        return relevance_score, None

# ---------------------------
# ---- 数据加载和预处理 ----
# ---------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df[df['score'].notna()]
df = df[df['score'] != 6]

print(f"Total essays: {len(df)} | unique prompts: {df['assignment'].nunique()}")
print("Score counts (before):")
print(df['score'].value_counts().sort_index())

# 构建prompt文本映射
prompt_texts = {}
for assignment in df['assignment'].unique():
    if assignment not in prompt_texts:
        prompt_texts[assignment] = f"Writing prompt: {assignment}"

print(f"Loaded {len(prompt_texts)} unique prompts")

# ========== 修复数据泄露问题 ==========
# 先划分数据集，再在训练集上进行平衡采样
print("\n=== 修复数据泄露问题 ===")
print("先划分数据集，再在训练集上进行平衡采样...")

# 按照7:1:2划分训练、验证、测试集
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['score'])
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['score'])

print(f"原始划分 - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 只在训练集上进行平衡采样
print("只在训练集上进行平衡采样...")
score_counts_train = train_df['score'].value_counts()
max_count_train = score_counts_train.max()

train_df_balanced = pd.concat([
    train_df[train_df['score'] == s].sample(max_count_train, replace=True, random_state=42)
    for s in sorted(score_counts_train.index)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# 使用平衡后的训练集
train_df = train_df_balanced

print("Score counts after balancing (training set only):")
print(f"Train: {train_df['score'].value_counts().sort_index()}")
print(f"Val: {val_df['score'].value_counts().sort_index()}")
print(f"Test: {test_df['score'].value_counts().sort_index()}")

print(f"最终数据集大小 - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 归一化目标 - 只在训练集上拟合
target_scaler = MinMaxScaler()
train_df['score_norm'] = target_scaler.fit_transform(train_df[['score']])
val_df['score_norm'] = target_scaler.transform(val_df[['score']])
test_df['score_norm'] = target_scaler.transform(test_df[['score']])

# 重置所有数据集的索引以确保一致性
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 预缓存图和特征
def precompute_all(df_src, dataset_name=""):
    graph_map = {}
    feature_map = {}
    prompt_map = {}
    
    for idx, row in tqdm(df_src.iterrows(), total=len(df_src), desc=f"Precomputing {dataset_name}"):
        text = str(row['full_text'])
        
        # 获取prompt文本
        prompt_name = row['prompt_name'] if pd.notna(row['prompt_name']) else row['assignment']
        prompt_text = prompt_texts.get(prompt_name, f"Writing prompt: {prompt_name}")
        
        # 缓存作文图和特征
        h = text_hash(text)
        graph_path = os.path.join(GRAPH_CACHE_DIR, f"{h}.pt")
        feature_path = os.path.join(FEATURE_CACHE_DIR, f"{h}.npy")
        
        if not os.path.exists(graph_path):
            g = build_graph_from_text(text)
            torch.save(g, graph_path)
        if not os.path.exists(feature_path):
            features = feature_extractor.extract_features(text)
            np.save(feature_path, features)
            
        # 使用重置后的索引作为键
        graph_map[idx] = graph_path
        feature_map[idx] = feature_path
        
        # 缓存prompt编码
        prompt_h = text_hash(prompt_text)
        prompt_path = os.path.join(PROMPT_CACHE_DIR, f"{prompt_h}.pt")
        if not os.path.exists(prompt_path):
            prompt_enc = tokenizer(
                prompt_text, 
                truncation=True, 
                padding='max_length', 
                max_length=128,
                return_tensors='pt'
            )
            torch.save(prompt_enc, prompt_path)
        prompt_map[idx] = prompt_path
            
    return graph_map, feature_map, prompt_map

print("Precomputing training data...")
train_graph_map, train_feature_map, train_prompt_map = precompute_all(train_df, "training data")
print("Precomputing validation data...")
val_graph_map, val_feature_map, val_prompt_map = precompute_all(val_df, "validation data")
print("Precomputing test data...")
test_graph_map, test_feature_map, test_prompt_map = precompute_all(test_df, "test data")

# 特征标准化 - 只在训练集上拟合，避免测试集泄露
all_train_features = []
for i in range(len(train_df)):
    if i in train_feature_map:
        features = np.load(train_feature_map[i])
        all_train_features.append(features)

if len(all_train_features) > 0:
    all_train_features = np.array(all_train_features)
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_train_features)
else:
    print("Warning: No training features found!")
    feature_scaler = StandardScaler()
    feature_scaler.fit(np.zeros((1, NUM_FEATURES)))

# ---------------------------
# ---- 数据集类 ----
# ---------------------------
class FusionGraphBertDataset(Dataset):
    def __init__(self, df, tokenizer, graph_map, feature_map, prompt_map, feature_scaler, is_training=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.graph_map = graph_map
        self.feature_map = feature_map
        self.prompt_map = prompt_map
        self.feature_scaler = feature_scaler
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['full_text'])
        
        # 数据增强 - 只在训练时使用
        if self.is_training and random.random() < 0.1:
            # 简单的数据增强：随机删除一些单词
            words = text.split()
            if len(words) > 10:
                # 随机删除5%的单词
                num_to_remove = max(1, int(len(words) * 0.05))
                indices_to_remove = random.sample(range(len(words)), num_to_remove)
                words = [word for i, word in enumerate(words) if i not in indices_to_remove]
                text = ' '.join(words)
        
        # BERT编码
        enc = self.tokenizer(text, truncation=True, padding='max_length', 
                           max_length=MAX_LENGTH, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        
        # 图数据
        if idx in self.graph_map:
            graph_path = self.graph_map[idx]
            graph = torch.load(graph_path, weights_only=False)
        else:
            graph = build_graph_from_text(text)
        
        # 特征数据
        if idx in self.feature_map:
            feature_path = self.feature_map[idx]
            features = np.load(feature_path)
        else:
            features = feature_extractor.extract_features(text)
        
        features = self.feature_scaler.transform(features.reshape(1, -1)).squeeze()
        features = torch.tensor(features, dtype=torch.float32)
        
        # Prompt编码
        if idx in self.prompt_map:
            prompt_path = self.prompt_map[idx]
            prompt_enc = torch.load(prompt_path, weights_only=False)
        else:
            prompt_name = row['prompt_name'] if pd.notna(row['prompt_name']) else row['assignment']
            prompt_text = prompt_texts.get(prompt_name, f"Writing prompt: {prompt_name}")
            prompt_enc = tokenizer(
                prompt_text, 
                truncation=True, 
                padding='max_length', 
                max_length=128,
                return_tensors='pt'
            )
        
        prompt_ids = prompt_enc['input_ids'].squeeze(0)
        prompt_mask = prompt_enc['attention_mask'].squeeze(0)
        
        score = torch.tensor(row['score_norm'], dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph": graph,
            "features": features,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "score": score
        }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = FusionGraphBertDataset(train_df, tokenizer, train_graph_map, train_feature_map, train_prompt_map, feature_scaler, is_training=True)
val_dataset = FusionGraphBertDataset(val_df, tokenizer, val_graph_map, val_feature_map, val_prompt_map, feature_scaler)
test_dataset = FusionGraphBertDataset(test_df, tokenizer, test_graph_map, test_feature_map, test_prompt_map, feature_scaler)

# 采样器 - 只在训练集上使用
score_to_count = train_df['score'].value_counts().to_dict()
sample_weights = [1.0 / score_to_count[s] for s in train_df['score']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch]).to(DEVICE)
    attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(DEVICE)
    
    # 图数据
    graphs = []
    for b in batch:
        g = b['graph']
        if g.x is None or g.x.size(0) == 0:
            g.x = torch.zeros((1, GRAPH_IN_DIM), dtype=torch.float32)
            g.edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        graphs.append(g)
    graph_batch = Batch.from_data_list(graphs).to(DEVICE)
    
    # 特征数据
    features = torch.stack([b['features'] for b in batch]).to(DEVICE)
    
    # Prompt数据
    prompt_ids = torch.stack([b['prompt_ids'] for b in batch]).to(DEVICE)
    prompt_mask = torch.stack([b['prompt_mask'] for b in batch]).to(DEVICE)
    
    scores = torch.stack([b['score'] for b in batch]).to(DEVICE)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graph": graph_batch,
        "features": features,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "score": scores
    }

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                         collate_fn=collate_fn, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       collate_fn=collate_fn, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)

# ---------------------------
# ---- 融合模型 ----
# ---------------------------
class RobustFeatureWeightingModule(nn.Module):
    """稳健的特征加权模块"""
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)  # 初始偏向1.0
    
    def forward(self, features):
        weights = self.attention(features)
        weighted_features = features * weights
        return weighted_features, weights

class GraphEncoder(nn.Module):
    """图编码器"""
    def __init__(self, in_dim=GRAPH_IN_DIM, hid=GRAPH_HIDDEN, out_dim=GRAPH_OUT, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hid)
        self.conv2 = GraphConv(hid, out_dim)
        self.conv3 = GraphConv(out_dim, out_dim // 2)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        x = self.dropout(F.relu(self.norm3(self.conv3(x, edge_index))))
        g = global_mean_pool(x, batch)
        return g

class FusionGraphBertAES(nn.Module):
    def __init__(self, model_name=MODEL_NAME, graph_out=GRAPH_OUT, 
                 num_features=NUM_FEATURES, hidden_dim=512, dropout=DROPOUT,
                 prompt_relevance_weight=PROMPT_RELEVANCE_WEIGHT):
        super().__init__()
        
        self.prompt_relevance_weight = prompt_relevance_weight
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size
        
        # 图编码器
        self.graph_encoder = GraphEncoder(in_dim=GRAPH_IN_DIM, hid=GRAPH_HIDDEN, 
                                        out_dim=graph_out, dropout=dropout)
        
        # 特征加权模块 - 使用稳健版本
        self.feature_weighting = RobustFeatureWeightingModule(num_features)
        
        # Prompt相关度模块 - 使用修复版本
        self.prompt_relevance = FixedPromptRelevanceModule(hidden_dim=bert_hidden, dropout=dropout)
        
        # 融合层
        self.fusion_input_dim = bert_hidden + graph_out // 2 + num_features
        print(f"🔧 Model initialization:")
        print(f"   - BERT hidden size: {bert_hidden}")
        print(f"   - Graph output size: {graph_out // 2}")
        print(f"   - Feature size: {num_features}")
        print(f"   - Prompt relevance weight: {prompt_relevance_weight}")
        print(f"   - Total fusion input: {self.fusion_input_dim}")
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 冻结部分BERT层
        if FREEZE_BERT_LAYERS > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < FREEZE_BERT_LAYERS:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, input_ids, attention_mask, graph: Batch, features, prompt_ids, prompt_mask):
        # BERT编码 - 作文
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        essay_embeddings = bert_out.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        bert_pooled = essay_embeddings[:, 0, :]  # 使用[CLS] token
        
        # BERT编码 - Prompt (需要梯度，因为相关度模块需要训练)
        prompt_out = self.bert(input_ids=prompt_ids, attention_mask=prompt_mask)
        prompt_embeddings = prompt_out.last_hidden_state  # [batch_size, prompt_len, hidden_dim]
        
        # 图编码
        graph_emb = self.graph_encoder(graph.x, graph.edge_index, graph.batch)
        
        # 特征加权
        weighted_features, feature_weights = self.feature_weighting(features)
        
        # Prompt相关度计算 - 修复版本
        relevance_score, _ = self.prompt_relevance(
            essay_embeddings, prompt_embeddings, attention_mask, prompt_mask
        )
        
        # 特征融合
        x = torch.cat([bert_pooled, graph_emb, weighted_features], dim=1)
        
        # 检查维度是否匹配
        if x.shape[1] != self.fusion_input_dim:
            raise ValueError(f"Fusion dimension mismatch: expected {self.fusion_input_dim}, got {x.shape[1]}")
        
        # 基础评分
        base_score = self.fusion_layers(x).squeeze(-1)
        
        # 结合Prompt相关度调整最终评分 - 使用更合理的调整方式
        # 相关度分数作为置信度权重
        relevance_weight = 1 + (relevance_score - 0.5) * self.prompt_relevance_weight * 50
        final_score = base_score * relevance_weight
        
        # 确保评分在合理范围内
        final_score = torch.clamp(final_score, 0.0, 1.0)
        
        return final_score, feature_weights, relevance_score, base_score
    
model = FusionGraphBertAES().to(DEVICE)

# ---------------------------
# ---- 优化器和损失函数 ----
# ---------------------------
class StableMultiTaskLoss(nn.Module):
    """稳定的多任务损失函数"""
    def __init__(self, alpha=0.95, beta=0.03, gamma=0.02):
        super().__init__()
        self.alpha = alpha  # 回归损失权重
        self.beta = beta    # 特征稀疏性损失权重
        self.gamma = gamma  # 相关度正则化损失权重
        self.regression_loss = nn.SmoothL1Loss()
        
    def forward(self, preds, targets, feature_weights, relevance_scores):
        reg_loss = self.regression_loss(preds, targets)
        
        # 更温和的特征稀疏性损失
        sparse_loss = torch.mean(torch.relu(0.05 - feature_weights))  # 鼓励权重至少为0.05
        
        # 更温和的相关度正则化：鼓励相关度分数在0.3-0.7之间
        relevance_mean = torch.mean(relevance_scores)
        # 使用更平滑的损失函数，避免过度惩罚
        relevance_loss = torch.abs(relevance_mean - 0.5)
        
        total_loss = self.alpha * reg_loss + self.beta * sparse_loss + self.gamma * relevance_loss
        
        return total_loss, reg_loss, sparse_loss, relevance_loss

# 优化器分组
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    # BERT参数
    {
        "params": [p for n, p in model.bert.named_parameters() if p.requires_grad],
        "lr": LR_BERT,
        "weight_decay": 0.01
    },
    # 图编码器参数
    {
        "params": [p for n, p in model.graph_encoder.named_parameters() if p.requires_grad],
        "lr": LR_GRAPH,
        "weight_decay": 0.01
    },
    # 特征加权参数
    {
        "params": [p for n, p in model.feature_weighting.named_parameters() if p.requires_grad],
        "lr": LR_FEATURES,
        "weight_decay": 0.001
    },
    # Prompt相关度参数 - 降低学习率，避免饱和
    {
        "params": [p for n, p in model.prompt_relevance.named_parameters() if p.requires_grad],
        "lr": LR_FEATURES * 0.5,  # 降低学习率
        "weight_decay": 0.001
    },
    # 融合层参数
    {
        "params": [p for n, p in model.fusion_layers.named_parameters() if p.requires_grad],
        "lr": LR_GRAPH,
        "weight_decay": 0.01
    },
]

optimizer = AdamW(optimizer_grouped_parameters)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
criterion = StableMultiTaskLoss()

# ---------------------------
# ---- 训练循环 ----
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 混合精度训练 - 更新为新的API
use_amp = DEVICE.type == "cuda" and torch.cuda.is_available()
if use_amp:
    amp_scaler = torch.cuda.amp.GradScaler()
    print("Using mixed precision training")
else:
    amp_scaler = None
    print("Mixed precision not available, using standard training")

early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path="fusion_graphbert_best.pth")

# 训练记录
train_loss_history, val_loss_history, val_qwk_history = [], [], []
best_val_loss = float("inf")

# 修复的健康检查函数
def detailed_health_check(model, batch, step):
    """修复的健康检查 - 避免维度不匹配错误"""
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph = batch["graph"]
        features = batch["features"]
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        
        # 前向传播到各模块
        bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        essay_embeddings = bert_out.last_hidden_state
        
        prompt_out = model.bert(input_ids=prompt_ids, attention_mask=prompt_mask)
        prompt_embeddings = prompt_out.last_hidden_state
        
        # 检查特征加权
        weighted_features, feature_weights = model.feature_weighting(features)
        
        # 检查相关度
        relevance_score, _ = model.prompt_relevance(
            essay_embeddings, prompt_embeddings, attention_mask, prompt_mask
        )
        
        # 检查基础评分 - 使用完整的模型前向传播
        _, _, relevance_scores, base_scores = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph=graph,
            features=features,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask
        )
        
        if step % 500 == 0:
            print(f"\n🔍 Step {step} - 详细健康检查:")
            print(f"   特征权重 - 均值: {feature_weights.mean().item():.6f}, 范围: [{feature_weights.min().item():.6f}, {feature_weights.max().item():.6f}]")
            print(f"   相关度分数 - 均值: {relevance_score.mean().item():.6f}, 范围: [{relevance_score.min().item():.6f}, {relevance_score.max().item():.6f}]")
            print(f"   基础评分 - 均值: {base_scores.mean().item():.6f}, 范围: [{base_scores.min().item():.6f}, {base_scores.max().item():.6f}]")
            print(f"   加权特征 - 均值: {weighted_features.mean().item():.6f}")
            
            # 检查梯度情况
            for name, param in model.prompt_relevance.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    if grad_mean > 0:
                        print(f"   {name} - 梯度均值: {grad_mean:.6f}")
    
    model.train()

print("🚀 Starting training with fixed modules...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    total_reg_loss = 0.0
    total_sparse_loss = 0.0
    total_relevance_loss = 0.0
    
    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
    
    for step, batch in pbar:
        # 定期检查模块健康状态
        if step % 500 == 0:
            detailed_health_check(model, batch, step)
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph = batch["graph"]
        features = batch["features"]
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        target = batch["score"]

        if use_amp:
            # 使用新的autocast API
            with torch.amp.autocast(device_type='cuda'):
                preds, feature_weights, relevance_scores, base_scores = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    graph=graph, 
                    features=features,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask
                )
                loss, reg_loss, sparse_loss, relevance_loss = criterion(preds, target, feature_weights, relevance_scores)
                loss = loss / ACCUM_STEPS

            amp_scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
        else:
            # 不使用混合精度
            preds, feature_weights, relevance_scores, base_scores = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                graph=graph, 
                features=features,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask
            )
            loss, reg_loss, sparse_loss, relevance_loss = criterion(preds, target, feature_weights, relevance_scores)
            loss = loss / ACCUM_STEPS

            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_train_loss += loss.item() * ACCUM_STEPS
        total_reg_loss += reg_loss.item() * ACCUM_STEPS
        total_sparse_loss += sparse_loss.item() * ACCUM_STEPS
        total_relevance_loss += relevance_loss.item() * ACCUM_STEPS
        
        # 显示相关度统计信息
        avg_relevance = relevance_scores.mean().item()
        std_relevance = relevance_scores.std().item()
        avg_feature_weights = feature_weights.mean().item()
        
        pbar.set_postfix({
            "total_loss": f"{(total_train_loss / (step+1)):.4f}",
            "reg_loss": f"{(total_reg_loss / (step+1)):.4f}",
            "sparse_loss": f"{(total_sparse_loss / (step+1)):.4f}",
            "rel_loss": f"{(total_relevance_loss / (step+1)):.4f}",
            "rel_mean": f"{avg_relevance:.3f}",
            "feat_w": f"{avg_feature_weights:.3f}"
        })

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # 验证
    model.eval()
    val_preds, val_true, val_relevance = [], [], []
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            graph = batch["graph"]
            features = batch["features"]
            prompt_ids = batch["prompt_ids"]
            prompt_mask = batch["prompt_mask"]
            target = batch["score"]

            if use_amp:
                # 使用新的autocast API
                with torch.amp.autocast(device_type='cuda'):
                    preds, feature_weights, relevance_scores, base_scores = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        graph=graph, 
                        features=features,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask
                    )
                    loss, _, _, _ = criterion(preds, target, feature_weights, relevance_scores)
            else:
                preds, feature_weights, relevance_scores, base_scores = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    graph=graph, 
                    features=features,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask
                )
                loss, _, _, _ = criterion(preds, target, feature_weights, relevance_scores)

            total_val_loss += loss.item()
            val_preds.extend(preds.detach().cpu().numpy())
            val_true.extend(target.detach().cpu().numpy())
            val_relevance.extend(relevance_scores.detach().cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    # 计算QWK
    min_score = df['score'].min()
    max_score = df['score'].max()
    
    val_true_denorm = (np.array(val_true) * (max_score - min_score)) + min_score
    val_pred_denorm = (np.array(val_preds) * (max_score - min_score)) + min_score

    val_true_round = np.round(val_true_denorm).astype(int)
    val_pred_round = np.clip(np.round(val_pred_denorm), min_score, max_score).astype(int)

    qwk = cohen_kappa_score(val_true_round, val_pred_round, weights='quadratic')
    val_qwk_history.append(qwk)

    # 计算相关度统计
    avg_relevance = np.mean(val_relevance)
    std_relevance = np.std(val_relevance)
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | QWK: {qwk:.4f}")
    print(f"Prompt Relevance - Mean: {avg_relevance:.4f} | Std: {std_relevance:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    # 早停
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("⛔ Early stopping triggered.")
        break

# ---------------------------
# ---- 评估和可视化 ----
# ---------------------------
print("\n--- Generating Final Evaluation ---\n")

# 加载最佳模型
model.load_state_dict(torch.load("fusion_graphbert_best.pth"))
model.eval()

# 验证集评估
final_val_preds, final_val_true, final_val_relevance = [], [], []
feature_weights_all = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph = batch["graph"]
        features = batch["features"]
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        target = batch["score"]
        
        if use_amp:
            # 使用新的autocast API
            with torch.amp.autocast(device_type='cuda'):
                preds, feature_weights, relevance_scores, base_scores = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    graph=graph, 
                    features=features,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask
                )
        else:
            preds, feature_weights, relevance_scores, base_scores = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                graph=graph, 
                features=features,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask
            )
        
        final_val_preds.extend(preds.detach().cpu().numpy())
        final_val_true.extend(target.detach().cpu().numpy())
        final_val_relevance.extend(relevance_scores.detach().cpu().numpy())
        feature_weights_all.extend(feature_weights.detach().cpu().numpy())

# 反归一化
final_y_true_val = (np.array(final_val_true) * (max_score - min_score)) + min_score
final_y_pred_continuous_val = (np.array(final_val_preds) * (max_score - min_score)) + min_score
final_y_pred_rounded_val = np.clip(final_y_pred_continuous_val.round(), min_score, max_score).astype(int)

# 计算验证集指标
final_mse_val = mean_squared_error(final_y_true_val, final_y_pred_continuous_val)
final_mae_val = mean_absolute_error(final_y_true_val, final_y_pred_continuous_val)
final_qwk_val = cohen_kappa_score(final_y_true_val.astype(int), final_y_pred_rounded_val, weights='quadratic')

print(f"\n=== Validation Set Evaluation Results ===")
print(f"QWK: {final_qwk_val:.4f}")
print(f"MSE: {final_mse_val:.4f}")
print(f"MAE: {final_mae_val:.4f}")
print(f"Prompt Relevance - Mean: {np.mean(final_val_relevance):.4f} | Std: {np.std(final_val_relevance):.4f}")

# 测试集评估
test_preds, test_true, test_relevance = [], [], []
test_feature_weights_all = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph = batch["graph"]
        features = batch["features"]
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        target = batch["score"]
        
        if use_amp:
            # 使用新的autocast API
            with torch.amp.autocast(device_type='cuda'):
                preds, feature_weights, relevance_scores, base_scores = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    graph=graph, 
                    features=features,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask
                )
        else:
            preds, feature_weights, relevance_scores, base_scores = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                graph=graph, 
                features=features,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask
            )
        
        test_preds.extend(preds.detach().cpu().numpy())
        test_true.extend(target.detach().cpu().numpy())
        test_relevance.extend(relevance_scores.detach().cpu().numpy())
        test_feature_weights_all.extend(feature_weights.detach().cpu().numpy())

# 反归一化
test_true_denorm = (np.array(test_true) * (max_score - min_score)) + min_score
test_pred_continuous = (np.array(test_preds) * (max_score - min_score)) + min_score
test_pred_rounded = np.clip(test_pred_continuous.round(), min_score, max_score).astype(int)

# 计算测试集指标
test_mse = mean_squared_error(test_true_denorm, test_pred_continuous)
test_mae = mean_absolute_error(test_true_denorm, test_pred_continuous)
test_qwk = cohen_kappa_score(test_true_denorm.astype(int), test_pred_rounded, weights='quadratic')

print(f"\n=== Test Set Evaluation Results ===")
print(f"QWK: {test_qwk:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"Prompt Relevance - Mean: {np.mean(test_relevance):.4f} | Std: {np.std(test_relevance):.4f}")

# 分析相关度与评分的关系
if len(test_relevance) > 1:
    relevance_corr = np.corrcoef(test_relevance, test_true_denorm)[0, 1]
    print(f"Correlation between relevance and true score: {relevance_corr:.4f}")
else:
    relevance_corr = 0
    print("Not enough samples to calculate correlation")

# 保存测试集预测结果
test_results_df = pd.DataFrame({
    'true_score': test_true_denorm,
    'pred_score_continuous': test_pred_continuous,
    'pred_score_rounded': test_pred_rounded,
    'prompt_relevance': test_relevance
})
test_results_df.to_csv('test_set_predictions.csv', index=False)
print(f"\nTest set predictions saved to: test_set_predictions.csv")

print(f"\n=== Final Results Summary ===")
print(f"Validation Set - QWK: {final_qwk_val:.4f}, MSE: {final_mse_val:.4f}, MAE: {final_mae_val:.4f}")
print(f"Test Set - QWK: {test_qwk:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
print(f"Prompt Relevance - Mean: {np.mean(test_relevance):.4f}, Std: {np.std(test_relevance):.4f}" if len(test_relevance) > 0 else "No relevance data")
print(f"Correlation between relevance and true score: {relevance_corr:.4f}")

print("\n🎯 Model training completed successfully!")

print("\n💾 Saving model and related files...")

# 定义保存路径
model_path = "fusion_graphbert_best.pth"
feature_scaler_path = "feature_scaler.pkl"
target_scaler_path = "target_scaler.pkl"
config_path = "model_config.json"
tokenizer_path = "./tokenizer"

# 1. 保存模型权重（已经通过早停保存，这里确保存在）
if os.path.exists(model_path):
    print(f"✓ Model weights saved at: {model_path}")
else:
    # 如果没有保存，重新保存一次
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model weights saved at: {model_path}")

# 2. 保存特征缩放器
try:
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    print(f"✓ Feature scaler saved at: {feature_scaler_path}")
except Exception as e:
    print(f"✗ Failed to save feature scaler: {e}")

# 3. 保存目标缩放器
try:
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"✓ Target scaler saved at: {target_scaler_path}")
except Exception as e:
    print(f"✗ Failed to save target scaler: {e}")

# 4. 保存模型配置
model_config = {
    "model_name": MODEL_NAME,
    "vocab_size": len(tokenizer),
    "fusion_input_dim": model.fusion_input_dim,
    "max_length": MAX_LENGTH,
    "graph_in_dim": GRAPH_IN_DIM,
    "graph_hidden": GRAPH_HIDDEN,
    "graph_out": GRAPH_OUT,
    "num_features": NUM_FEATURES,
    "dropout": DROPOUT,
    "freeze_bert_layers": FREEZE_BERT_LAYERS,
    "prompt_relevance_weight": PROMPT_RELEVANCE_WEIGHT,
    "score_min": float(target_scaler.data_min_[0]),
    "score_max": float(target_scaler.data_max_[0]),
    "feature_names": [
        "char_count", "word_count", "comma_count", "unique_words", 
        "proper_nouns", "determiners", "nouns", "adverbs", 
        "adjectives", "prepositions", "gunning_fog", "smog_index", 
        "rix_index", "dale_chall", "word_types", "sentence_count", 
        "long_words", "complex_words", "non_common_words", "difficult_words"
    ],
    "training_info": {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "epochs_trained": epoch,
        "best_val_qwk": float(final_qwk_val),
        "test_qwk": float(test_qwk),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
}

try:
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    print(f"✓ Model config saved at: {config_path}")
except Exception as e:
    print(f"✗ Failed to save model config: {e}")

# 5. 保存tokenizer
try:
    # 确保tokenizer目录存在
    os.makedirs(tokenizer_path, exist_ok=True)
    
    # 保存tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    print(f"✓ Tokenizer saved at: {tokenizer_path}")
    
    # 验证tokenizer可以重新加载
    test_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("✓ Tokenizer reload test successful")
except Exception as e:
    print(f"✗ Failed to save tokenizer: {e}")

# 6. 保存完整的模型（可选，用于推理）
try:
    # 保存完整模型结构（需要同时保存模型类和状态）
    full_model_path = "fusion_graphbert_full.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }, full_model_path)
    print(f"✓ Full model package saved at: {full_model_path}")
except Exception as e:
    print(f"✗ Failed to save full model package: {e}")

# 7. 保存训练历史
training_history = {
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "val_qwk": val_qwk_history
}

try:
    history_path = "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Training history saved at: {history_path}")
except Exception as e:
    print(f"✗ Failed to save training history: {e}")

print("\n📁 All model files have been saved successfully!")
print("Files created:")
print(f"  - Model weights: {model_path}")
print(f"  - Feature scaler: {feature_scaler_path}")
print(f"  - Target scaler: {target_scaler_path}")
print(f"  - Model config: {config_path}")
print(f"  - Tokenizer: {tokenizer_path}/")
print(f"  - Full model package: fusion_graphbert_full.pth")
print(f"  - Training history: training_history.json")
print(f"  - Test predictions: test_set_predictions.csv")
