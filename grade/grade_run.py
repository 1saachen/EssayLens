import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_mean_pool
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
import spacy
import nltk
import re
from textstat.textstat import textstat
from collections import Counter
import hashlib
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

if __package__:
    from .model_config import read_model_config
else:
    from model_config import read_model_config

# ---------------------------
# ---- 从配置文件读取配置 ----
# ---------------------------
def load_config(config_path="model_config.json"):
    """从配置文件加载所有配置"""
    config = read_model_config(config_path)
    
    # 设置全局变量
    global MODEL_NAME, MAX_LENGTH, GRAPH_IN_DIM, GRAPH_HIDDEN, GRAPH_OUT
    global NUM_FEATURES, PROMPT_RELEVANCE_WEIGHT, DROPOUT, FREEZE_BERT_LAYERS
    global MIN_SCORE, MAX_SCORE
    
    MODEL_NAME = config.get("model_name", "roberta-base")
    MAX_LENGTH = config.get("max_length", 256)
    GRAPH_IN_DIM = config.get("graph_in_dim", 2)
    GRAPH_HIDDEN = config.get("graph_hidden", 64)
    GRAPH_OUT = config.get("graph_out", 256)
    NUM_FEATURES = config.get("num_features", 20)
    PROMPT_RELEVANCE_WEIGHT = config.get("prompt_relevance_weight", 0.15)
    DROPOUT = config.get("dropout", 0.3)
    FREEZE_BERT_LAYERS = config.get("freeze_bert_layers", 6)
    MIN_SCORE = config.get("score_min", 1.0)
    MAX_SCORE = config.get("score_max", 5.0)
    
    return config

# 加载配置
config = load_config()

print("🔧 配置信息:")
print(f"  - 模型: {MODEL_NAME}")
print(f"  - 分数范围: {MIN_SCORE}-{MAX_SCORE}")
print(f"  - 图编码器输出: {GRAPH_OUT}")
print(f"  - 特征数量: {NUM_FEATURES}")
print(f"  - Prompt相关度权重: {PROMPT_RELEVANCE_WEIGHT}")
print(f"  - Dropout: {DROPOUT}")

# ---------------------------
# ---- 与训练代码完全一致的特征提取器 ----
# ---------------------------
class FeatureExtractor:
    def __init__(self):
        self.dale_chall_common_words = self._load_dale_chall_words()
        self.NUM_FEATURES = NUM_FEATURES
        self.feature_names = [
            'char_count', 'word_count', 'comma_count', 'unique_words',
            'proper_nouns', 'determiners', 'nouns', 'adverbs', 'adjectives', 'prepositions',
            'gunning_fog', 'smog_index', 'rix_index', 'dale_chall', 'word_types',
            'sentence_count', 'long_words', 'complex_words', 'uncommon_words', 'difficult_words'
        ]
    
    def _load_dale_chall_words(self):
        """加载常用词表 - 与训练代码完全一致"""
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
        """安全的分词函数 - 与训练代码完全一致"""
        try:
            return nltk.word_tokenize(str(text))
        except:
            try:
                return str(text).split()
            except:
                return []
    
    def extract_features(self, text):
        """提取20维语言学特征 - 与训练代码完全一致"""
        if pd.isna(text) or text == "" or text is None:
            return np.zeros(self.NUM_FEATURES, dtype=float), {}
        
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
            
            # 创建特征详细字典
            feature_details = dict(zip(self.feature_names, feats))
            
            return np.array(feats, dtype=float), feature_details
            
        except Exception as e:
            print(f"Error computing features: {e}")
            return np.zeros(self.NUM_FEATURES, dtype=float), {name: 0 for name in self.feature_names}

# ---------------------------
# ---- 与训练代码完全一致的图构建函数 ----
# ---------------------------
def build_graph_from_text(text: str, nlp, max_nodes: int = 256) -> Data:
    """构建依赖图 - 与训练代码完全一致"""
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
    
    # 计算图结构指标
    graph_metrics = {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'avg_dependency_distance': np.mean([abs(edge[0] - edge[1]) for edge in edges]) if edges else 0,
        'root_nodes': sum(is_root),
        'syntactic_complexity': len(edges) / max(len(nodes), 1)
    }
    
    return Data(x=x, edge_index=edge_index), graph_metrics

# ---------------------------
# ---- 与训练代码完全一致的模型组件 ----
# ---------------------------
class FixedPromptRelevanceModule(nn.Module):
    """修复的Prompt相关度评估模块 - 与训练代码完全一致"""
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

class RobustFeatureWeightingModule(nn.Module):
    """稳健的特征加权模块 - 与训练代码完全一致"""
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
    """图编码器 - 与训练代码完全一致"""
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
                 prompt_relevance_weight=PROMPT_RELEVANCE_WEIGHT, 
                 vocab_size=None, fusion_input_dim=None):
        super().__init__()
        
        self.prompt_relevance_weight = prompt_relevance_weight
        
        # BERT编码器 - 使用训练时的模型名称
        print(f"🔧 加载BERT模型: {model_name}")
        try:
            # 尝试使用DebertaV2Tokenizer的方式加载
            from transformers import DebertaV2Tokenizer
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            print("✓ DebertaV2Tokenizer (slow) loaded successfully")
        except Exception as e:
            print(f"DebertaV2Tokenizer failed: {e}")
            try:
                # 如果失败，尝试使用AutoTokenizer但禁用fast版本
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                print("✓ AutoTokenizer with use_fast=False loaded")
            except Exception as e2:
                print(f"All tokenizer attempts failed: {e2}")
                # 最后回退到其他模型
                print("Falling back to roberta-base...")
                model_name = "roberta-base"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 如果提供了词汇表大小，调整embedding层
        if vocab_size is not None and vocab_size != self.bert.embeddings.word_embeddings.weight.shape[0]:
            print(f"🔄 调整词汇表大小: {self.bert.embeddings.word_embeddings.weight.shape[0]} -> {vocab_size}")
            old_embedding = self.bert.embeddings.word_embeddings
            new_embedding = nn.Embedding(vocab_size, old_embedding.embedding_dim)
            
            # 复制已有的权重，新的token用随机初始化
            min_size = min(vocab_size, old_embedding.weight.shape[0])
            new_embedding.weight.data[:min_size] = old_embedding.weight.data[:min_size]
            
            self.bert.embeddings.word_embeddings = new_embedding
        
        bert_hidden = self.bert.config.hidden_size
        print(f"📊 BERT隐藏层维度: {bert_hidden}")
        
        # 冻结BERT层 - 与训练代码完全一致
        if FREEZE_BERT_LAYERS > 0:
            print(f"🔒 冻结前 {FREEZE_BERT_LAYERS} 层BERT参数")
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < FREEZE_BERT_LAYERS:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # 图编码器 - 与训练代码完全一致
        self.graph_encoder = GraphEncoder(in_dim=GRAPH_IN_DIM, hid=GRAPH_HIDDEN, 
                                        out_dim=graph_out, dropout=dropout)
        
        # 特征加权模块 - 使用稳健版本
        self.feature_weighting = RobustFeatureWeightingModule(num_features)
        
        # Prompt相关度模块 - 使用修复版本
        self.prompt_relevance = FixedPromptRelevanceModule(hidden_dim=bert_hidden, dropout=dropout)
        
        # 融合层
        if fusion_input_dim is None:
            # 计算融合输入维度：BERT隐藏层 + 图编码器输出 + 特征数
            # 图编码器输出是 GRAPH_OUT // 2，因为第三层输出维度减半
            graph_output_dim = GRAPH_OUT // 2
            self.fusion_input_dim = bert_hidden + graph_output_dim + num_features
        else:
            self.fusion_input_dim = fusion_input_dim
            
        print(f"🔧 融合输入维度: {self.fusion_input_dim} (BERT:{bert_hidden} + 图:{GRAPH_OUT//2} + 特征:{num_features})")
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_ids, attention_mask, graph: Batch, features, prompt_ids, prompt_mask):
        # BERT编码 - 作文
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        essay_embeddings = bert_out.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        bert_pooled = essay_embeddings[:, 0, :]  # 使用[CLS] token
        
        # BERT编码 - Prompt (需要梯度，因为相关度模块需要训练) - 与训练代码完全一致
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
            print(f"⚠ 维度不匹配: 期望 {self.fusion_input_dim}, 实际 {x.shape[1]}")
            print(f"   各组件维度: BERT({bert_pooled.shape}), Graph({graph_emb.shape}), Features({weighted_features.shape})")
            raise ValueError(f"融合层维度不匹配: 期望 {self.fusion_input_dim}, 实际 {x.shape[1]}")
        
        # 基础评分
        base_score = self.fusion_layers(x).squeeze(-1)
        
        # 结合Prompt相关度调整最终评分 - 与训练代码完全一致
        # 相关度分数作为置信度权重
        relevance_weight = 1 + (relevance_score - 0.5) * self.prompt_relevance_weight * 50
        final_score = base_score * relevance_weight
        
        # 确保评分在合理范围内
        final_score = torch.clamp(final_score, 0.0, 1.0)
        
        # 返回更多中间结果用于分析
        intermediate_outputs = {
            'bert_pooled': bert_pooled,
            'graph_embedding': graph_emb,
            'weighted_features': weighted_features,
            'feature_weights': feature_weights,
            'relevance_score': relevance_score,
            'base_score': base_score,
            'relevance_weight': relevance_weight
        }
        
        return final_score, intermediate_outputs

# ---------------------------
# ---- 增强的评分器 ----
# ---------------------------
class EssayScorer:
    def __init__(self, model_path="fusion_graphbert_best.pth", 
                 feature_scaler_path="feature_scaler.pkl",
                 target_scaler_path="target_scaler.pkl",
                 config_path="model_config.json",
                 tokenizer_path="./tokenizer",
                 device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 首先加载配置
        self.model_config = self._load_model_config(config_path, tokenizer_path)
        
        # 然后加载组件（使用配置中的参数）
        self._load_components(tokenizer_path)
        self._load_scalers(feature_scaler_path, target_scaler_path)
        self._load_model(model_path)
        
        print("✓ Essay Scorer initialized successfully")
    
    def _load_model_config(self, config_path, tokenizer_path="./tokenizer"):
        """加载模型配置"""
        if os.path.exists(config_path):
            config = load_config(config_path)
            print(f"✓ Model config loaded from {config_path}")
            
            # 确保配置中有必要的字段
            if "vocab_size" not in config:
                # 从tokenizer目录推断词汇表大小
                if os.path.exists(tokenizer_path):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                        config["vocab_size"] = len(tokenizer)
                        print(f"📚 从tokenizer推断词汇表大小: {config['vocab_size']}")
                    except:
                        pass  # _load_model uses the actual loaded tokenizer length.
            
            if "fusion_input_dim" not in config:
                # 计算融合输入维度
                model_name = config.get("model_name", "roberta-base")
                try:
                    bert_config = AutoConfig.from_pretrained(model_name)
                    bert_hidden = bert_config.hidden_size
                    # 图编码器输出维度是 GRAPH_OUT // 2
                    graph_out = config.get("graph_out", 256) // 2
                    num_features = config.get("num_features", 20)
                    config["fusion_input_dim"] = bert_hidden + graph_out + num_features
                    print(f"🔧 计算融合输入维度: {config['fusion_input_dim']}")
                except:
                    config["fusion_input_dim"] = 768 + 128 + 20  # 默认值
                    print(f"⚠ 使用默认融合输入维度: {config['fusion_input_dim']}")
            
            return config
        else:
            print(f"❌ Model config file {config_path} not found.")
            raise FileNotFoundError(f"Model config file {config_path} not found")
    
    def _load_components(self, tokenizer_path):
        """加载所有必要的组件 - 使用保存的tokenizer"""
        # 加载保存的tokenizer
        print("Loading tokenizer...")
        if os.path.exists(tokenizer_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ Tokenizer loaded from {tokenizer_path}")
                print(f"📚 Tokenizer vocab size: {len(self.tokenizer)}")
            except Exception as e:
                print(f"❌ Error loading tokenizer: {e}")
                raise
        else:
            print(f"❌ Tokenizer path {tokenizer_path} not found.")
            raise FileNotFoundError(f"Tokenizer path {tokenizer_path} not found")
        
        # 加载spaCy - 与训练代码完全一致
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
            print("✓ spaCy model loaded")
        except OSError:
            print("Downloading spaCy English model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        
        # 初始化特征提取器 - 与训练代码完全一致
        self.feature_extractor = FeatureExtractor()
        
        # 下载nltk数据 - 与训练代码完全一致
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    
    def _load_scalers(self, feature_scaler_path, target_scaler_path):
        """加载特征标准化器和目标分数归一化器"""
        # 加载特征标准化器
        try:
            if os.path.exists(feature_scaler_path):
                with open(feature_scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✓ Feature scaler loaded from {feature_scaler_path}")
            else:
                print(f"❌ Feature scaler file {feature_scaler_path} not found.")
                raise FileNotFoundError(f"Feature scaler file {feature_scaler_path} not found")
        except Exception as e:
            print(f"❌ Error loading feature scaler: {e}")
            raise
        
        # 加载目标分数归一化器
        try:
            if os.path.exists(target_scaler_path):
                with open(target_scaler_path, 'rb') as f:
                    self.target_scaler = pickle.load(f)
                print(f"✓ Target scaler loaded from {target_scaler_path}")
                # 检查scaler类型
                print(f"📊 Target scaler type: {type(self.target_scaler)}")
            else:
                print(f"❌ Target scaler file {target_scaler_path} not found.")
                raise FileNotFoundError(f"Target scaler file {target_scaler_path} not found")
        except Exception as e:
            print(f"❌ Error loading target scaler: {e}")
            raise
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print("Loading trained model...")
        
        # 使用配置中的参数创建模型
        model_name = self.model_config.get("model_name", "roberta-base")
        vocab_size = self.model_config.get("vocab_size", len(self.tokenizer))
        fusion_input_dim = self.model_config.get("fusion_input_dim")
        
        print(f"🔧 创建模型: {model_name}, vocab_size={vocab_size}, fusion_input_dim={fusion_input_dim}")
        
        self.model = FusionGraphBertAES(
            model_name=model_name,
            graph_out=self.model_config.get("graph_out", 256),
            num_features=self.model_config.get("num_features", 20),
            dropout=self.model_config.get("dropout", 0.3),
            prompt_relevance_weight=self.model_config.get("prompt_relevance_weight", 0.15),
            vocab_size=vocab_size,
            fusion_input_dim=fusion_input_dim
        ).to(self.device)
        
        # 加载训练好的权重
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                # 严格加载，确保所有参数匹配
                self.model.load_state_dict(state_dict, strict=True)
                print(f"✓ Model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
        else:
            print(f"❌ Model file {model_path} not found.")
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        self.model.eval()
        print("✓ Model set to evaluation mode")
    
    def preprocess_essay(self, essay_text, prompt_text):
        """预处理单篇作文 - 与训练代码完全一致"""
        # BERT编码作文
        essay_enc = self.tokenizer(
            essay_text, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LENGTH, 
            return_tensors='pt'
        )
        
        # BERT编码Prompt
        prompt_enc = self.tokenizer(
            prompt_text, 
            truncation=True, 
            padding='max_length', 
            max_length=128, 
            return_tensors='pt'
        )
        
        # 构建图
        graph, graph_metrics = build_graph_from_text(essay_text, self.nlp)
        
        # 提取特征并标准化
        features, feature_details = self.feature_extractor.extract_features(essay_text)
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1)).squeeze()
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        return {
            "input_ids": essay_enc['input_ids'].squeeze(0),
            "attention_mask": essay_enc['attention_mask'].squeeze(0),
            "graph": graph,
            "graph_metrics": graph_metrics,
            "features": features_tensor,
            "features_raw": features,
            "feature_details": feature_details,
            "prompt_ids": prompt_enc['input_ids'].squeeze(0),
            "prompt_mask": prompt_enc['attention_mask'].squeeze(0)
        }
    
    def _denormalize_score(self, normalized_score):
        """反归一化分数 - 修复了GradScaler错误"""
        try:
            # 确保输入是标量值，而不是张量
            if torch.is_tensor(normalized_score):
                normalized_score = normalized_score.item()
            
            # 检查target_scaler类型
            if hasattr(self.target_scaler, 'inverse_transform'):
                # 如果是sklearn的scaler
                score_np = np.array([[normalized_score]])
                denormalized = self.target_scaler.inverse_transform(score_np)[0][0]
                return float(denormalized)
            else:
                # 默认使用MinMax归一化
                min_score = self.model_config.get("score_min", 1.0)
                max_score = self.model_config.get("score_max", 5.0)
                denormalized = normalized_score * (max_score - min_score) + min_score
                return float(denormalized)
        except Exception as e:
            print(f"⚠ 反归一化失败: {e}，使用默认方法")
            min_score = self.model_config.get("score_min", 1.0)
            max_score = self.model_config.get("score_max", 5.0)
            denormalized = normalized_score * (max_score - min_score) + min_score
            return float(denormalized)
    
    def _analyze_model_outputs(self, intermediate_outputs):
        """分析模型的中间输出，提取多维特征指标"""
        analysis = {}
        
        # BERT embedding 分析
        bert_pooled = intermediate_outputs['bert_pooled']
        analysis['bert_embedding_norm'] = float(torch.norm(bert_pooled).item())
        analysis['bert_embedding_mean'] = float(torch.mean(bert_pooled).item())
        analysis['bert_embedding_std'] = float(torch.std(bert_pooled).item())
        
        # Graph embedding 分析
        graph_emb = intermediate_outputs['graph_embedding']
        analysis['graph_embedding_norm'] = float(torch.norm(graph_emb).item())
        analysis['graph_embedding_mean'] = float(torch.mean(graph_emb).item())
        
        # 特征权重分析
        feature_weights = intermediate_outputs['feature_weights']
        analysis['feature_weights_mean'] = float(torch.mean(feature_weights).item())
        analysis['feature_weights_std'] = float(torch.std(feature_weights).item())
        analysis['feature_weights_max'] = float(torch.max(feature_weights).item())
        analysis['feature_weights_min'] = float(torch.min(feature_weights).item())
        
        # 相关度分析
        analysis['relevance_score'] = float(intermediate_outputs['relevance_score'].item())
        analysis['relevance_weight'] = float(intermediate_outputs['relevance_weight'].item())
        
        return analysis
    
    def score_single_essay(self, essay_text, prompt_text, return_details=True):
        """对单篇作文进行评分，返回多维特征指标"""
        try:
            print(f"📝 开始评分作文 (词数: {len(essay_text.split())})")
            
            # 预处理
            data = self.preprocess_essay(essay_text, prompt_text)
            
            # 准备批量数据
            input_ids = data["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = data["attention_mask"].unsqueeze(0).to(self.device)
            graph_batch = Batch.from_data_list([data["graph"]]).to(self.device)
            features = data["features"].unsqueeze(0).to(self.device)
            prompt_ids = data["prompt_ids"].unsqueeze(0).to(self.device)
            prompt_mask = data["prompt_mask"].unsqueeze(0).to(self.device)
            
            print("🧠 运行模型推理...")
            # 模型预测
            with torch.no_grad():
                final_score, intermediate_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph=graph_batch,
                    features=features,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask
                )
            
            # 使用修复的反归一化方法
            final_score_denorm = self._denormalize_score(final_score)
            base_score_denorm = self._denormalize_score(intermediate_outputs['base_score'])
            
            # 分析模型中间输出
            model_analysis = self._analyze_model_outputs(intermediate_outputs)
            
            # 构建详细结果
            score_details = {
                "final_score": round(final_score_denorm, 2),
                "base_score": round(base_score_denorm, 2),
                "prompt_relevance": round(model_analysis['relevance_score'], 3),
                "adjusted_by_relevance": round((final_score_denorm - base_score_denorm), 2),
                "word_count": len(essay_text.split()),
                "char_count": len(essay_text),
                
                # 模型内部特征
                "model_features": {
                    "bert_embedding": {
                        "norm": round(model_analysis['bert_embedding_norm'], 4),
                        "mean": round(model_analysis['bert_embedding_mean'], 4),
                        "std": round(model_analysis['bert_embedding_std'], 4)
                    },
                    "graph_embedding": {
                        "norm": round(model_analysis['graph_embedding_norm'], 4),
                        "mean": round(model_analysis['graph_embedding_mean'], 4)
                    },
                    "feature_weights": {
                        "mean": round(model_analysis['feature_weights_mean'], 4),
                        "std": round(model_analysis['feature_weights_std'], 4),
                        "max": round(model_analysis['feature_weights_max'], 4),
                        "min": round(model_analysis['feature_weights_min'], 4)
                    },
                    "relevance_metrics": {
                        "score": round(model_analysis['relevance_score'], 4),
                        "weight": round(model_analysis['relevance_weight'], 4)
                    }
                },
                
                # 语言特征
                "linguistic_features": data["feature_details"],
                
                # 图结构特征
                "graph_structure": data["graph_metrics"],
                
                # 原始特征值（标准化前）
                "raw_features": dict(zip(self.feature_extractor.feature_names, 
                                       [round(x, 2) for x in data["features_raw"]]))
            }
            
            print(f"✅ 评分完成: {final_score_denorm:.2f}")
            
            return score_details
                
        except Exception as e:
            print(f"❌ Error scoring essay: {e}")
            import traceback
            traceback.print_exc()
            
            error_result = {
                "final_score": None,
                "base_score": None,
                "prompt_relevance": None,
                "adjusted_by_relevance": None,
                "word_count": len(essay_text.split()),
                "char_count": len(essay_text),
                "error": str(e),
                "model_features": None,
                "linguistic_features": None,
                "graph_structure": None,
                "raw_features": None
            }
            return error_result
    
    def score_multiple_essays(self, essays_data):
        """
        对多篇作文进行批量评分
        
        Args:
            essays_data: list of dict, 每个dict包含:
                - 'essay_text': 作文文本
                - 'prompt_text': 题目文本
                - 'essay_id': 可选，作文ID
        
        Returns:
            list of dict: 评分结果
        """
        results = []
        
        for i, essay_data in enumerate(essays_data):
            essay_text = essay_data['essay_text']
            prompt_text = essay_data['prompt_text']
            essay_id = essay_data.get('essay_id', f"essay_{i+1}")
            
            print(f"\n{'='*50}")
            print(f"评分作文 {essay_id}")
            print(f"{'='*50}")
            
            score_details = self.score_single_essay(essay_text, prompt_text)
            
            result = {
                'essay_id': essay_id,
                'final_score': score_details['final_score'],
                'base_score': score_details['base_score'],
                'prompt_relevance': score_details['prompt_relevance'],
                'adjusted_by_relevance': score_details['adjusted_by_relevance'],
                'word_count': score_details['word_count'],
                'char_count': score_details['char_count'],
                'model_features': score_details['model_features'],
                'linguistic_features': score_details['linguistic_features'],
                'graph_structure': score_details['graph_structure'],
                'raw_features': score_details['raw_features']
            }
            
            if 'error' in score_details:
                result['error'] = score_details['error']
            
            results.append(result)
            
            if score_details['final_score'] is not None:
                print(f"✅ {essay_id}: 最终分数 {score_details['final_score']} | 相关度 {score_details['prompt_relevance']}")
                print(f"   BERT嵌入范数: {score_details['model_features']['bert_embedding']['norm']}")
                print(f"   图嵌入范数: {score_details['model_features']['graph_embedding']['norm']}")
                print(f"   特征权重均值: {score_details['model_features']['feature_weights']['mean']}")
            else:
                print(f"❌ {essay_id}: 评分失败 - {score_details.get('error', 'Unknown error')}")
        
        return results

# ---------------------------
# ---- 使用示例 ----
# ---------------------------
def main():
    try:
        # 初始化评分器
        scorer = EssayScorer(
            model_path="fusion_graphbert_best.pth",
            feature_scaler_path="feature_scaler.pkl",
            target_scaler_path="target_scaler.pkl",
            config_path="model_config.json",
            tokenizer_path="./tokenizer"
        )
        
        # 测试评分
        prompt_text = """Environmental Protection: A Shared Responsibility"""
        essay_text = """

In the modern era, environmental protection has emerged as an increasingly urgent global issue that demands our immediate and collective attention. The Earth, our only home in the vast universe, is facing a multitude of environmental challenges that pose a grave threat to the very existence of life as we know it.

One of the most pressing problems is air pollution. Factories, with their towering chimneys, emit large amounts of harmful gases such as sulfur dioxide and nitrogen oxides into the atmosphere. Meanwhile, the ever - increasing number of vehicles on the road continuously releases exhaust fumes. These pollutants not only cause a range of respiratory diseases in humans, including asthma and lung cancer, but also lead to the formation of smog. Smog blankets cities, reducing visibility and disrupting daily life, from traffic congestion to flight delays.

Water pollution is another significant concern that cannot be ignored. Industrial waste, often containing toxic chemicals, agricultural runoff laden with pesticides and fertilizers, and untreated sewage are frequently discharged into rivers and oceans. This contamination kills a vast number of aquatic organisms, disrupts the balance of marine ecosystems, and makes the water unfit for human consumption and recreational activities.

To address these pressing issues, we all have a crucial role to play. Governments should take the lead by enacting and strictly enforcing comprehensive environmental laws and regulations. They can impose limits on pollution emissions from industries and vehicles, and offer incentives for companies to adopt cleaner production technologies. Additionally, investing in renewable energy sources like solar and wind power is essential to reduce our reliance on fossil fuels, which are major contributors to environmental degradation.

As individuals, we can make simple yet effective changes in our daily lives. We can practice the three Rs - reduce, reuse, and recycle - to minimize waste generation. Choosing to walk, bike, or use public transportation instead of driving alone can significantly cut down on air pollution. Moreover, conserving water by fixing leaks and using it wisely is of great importance.

In conclusion, environmental protection is not a one - person job but a shared responsibility that transcends national boundaries. Only by working together at all levels, from governments to individuals, can we safeguard our planet and ensure a sustainable and healthy future for generations to come.
      """
        
        score_details = scorer.score_single_essay(essay_text, prompt_text)
        
        print("\n📊 详细评分结果:")
        print(f"\n🎯 评分指标:")
        print(f"  最终分数: {score_details['final_score']}")
        print(f"  基础分数: {score_details['base_score']}")
        print(f"  Prompt相关度: {score_details['prompt_relevance']}")
        print(f"  相关度调整: {score_details['adjusted_by_relevance']}")
        
        print(f"\n📝 文本统计:")
        print(f"  词数: {score_details['word_count']}")
        print(f"  字符数: {score_details['char_count']}")
        
        print(f"\n🧠 模型内部特征:")
        mf = score_details['model_features']
        print(f"  BERT嵌入 - 范数: {mf['bert_embedding']['norm']}, 均值: {mf['bert_embedding']['mean']}, 标准差: {mf['bert_embedding']['std']}")
        print(f"  图嵌入 - 范数: {mf['graph_embedding']['norm']}, 均值: {mf['graph_embedding']['mean']}")
        print(f"  特征权重 - 均值: {mf['feature_weights']['mean']}, 标准差: {mf['feature_weights']['std']}")
        print(f"  相关度指标 - 分数: {mf['relevance_metrics']['score']}, 权重: {mf['relevance_metrics']['weight']}")
        
        print(f"\n📈 语言特征 (前5个最重要的):")
        linguistic = score_details['linguistic_features']
        # 按值排序显示最重要的特征
        sorted_features = sorted(linguistic.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for name, value in sorted_features:
            print(f"  {name}: {value:.2f}")
        
        print(f"\n🕸️ 图结构特征:")
        graph = score_details['graph_structure']
        for key, value in graph.items():
            print(f"  {key}: {value:.2f}")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()