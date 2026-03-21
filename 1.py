import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import pickle
from io import StringIO
from torch.utils.data import DataLoader, TensorDataset
import esm
import time
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

# ==========================================
# 模型架构定义
# ==========================================
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    st.warning("Mamba模块未安装，将使用替代实现")

    # 简单的替代实现，仅用于演示
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            return self.norm(x)


class CNNBranch(nn.Module):
    def __init__(self, input_dim=480, num_classes=2):  # input_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Unflatten(1, (1, 256)),
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.classifier(feat)


class TransformerBranch(nn.Module):
    def __init__(self, input_dim=480, d_model=256, nhead=8, num_classes=2):  # input_dim
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)


class MambaBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.preprocess = nn.Linear(input_dim, 256)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=256, d_state=16, d_conv=4, expand=2) for _ in range(5)
        ])
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(1)
        for block in self.mamba_blocks:
            x = x + block(x)
        x = self.norm(x).squeeze(1)
        return self.classifier(x)


class MutualLearningModel(nn.Module):
    def __init__(self, input_dim=480, num_classes=2, embed_dim=128):  # input_dim
        super().__init__()
        self.cnn = CNNBranch(input_dim, num_classes)
        self.trans = TransformerBranch(input_dim, num_classes=num_classes)
        self.mamba = MambaBranch(input_dim, num_classes)
        self.logits_norm = nn.LayerNorm(num_classes)
        self.feature_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm2 = nn.LayerNorm(embed_dim)
        total_gate_dim = embed_dim * 3 + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(total_gate_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.8 + 1e-6)))
        self.refine = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LayerNorm(num_classes),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        o1, o2, o3 = self.cnn(x), self.trans(x), self.mamba(x)
        branches = torch.stack([o1, o2, o3], dim=1)
        branches_norm = self.logits_norm(branches)
        x_proj = self.feature_proj(branches_norm)
        attn_out, _ = self.attn1(x_proj, x_proj, x_proj)
        x = self.attn_norm1(x_proj + attn_out)
        x = self.ffn_norm1(x + self.ffn1(x))
        attn_out, _ = self.attn2(x, x, x)
        x = self.attn_norm2(x + attn_out)
        x = self.ffn_norm2(x + self.ffn2(x))
        raw_logits = branches.flatten(1)
        fused_proj = x.flatten(1)
        combined_feat = torch.cat([fused_proj, raw_logits], dim=1)
        gate_scores = self.gate(combined_feat)
        temp = F.softplus(self.log_temp) + 1e-4
        weights = F.softmax(gate_scores / temp, dim=1).unsqueeze(-1)
        o_fused = (branches * weights).sum(dim=1)
        o_fused = o_fused + self.refine(o_fused)
        return o1, o2, o3, o_fused


# ==========================================
# 特征提取类
# ==========================================
class ESMFeatureExtractor:
    def __init__(self):
        self.gpu_model = None
        self.gpu_batch_converter = None
        self.cpu_model = None
        self.cpu_batch_converter = None
        self.device = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            if torch.cuda.is_available():
                print("🚀 尝试加载GPU模型（ESM-2 35M）...")
                self.gpu_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()  # 替换为 35M 模型
                self.gpu_device = torch.device('cuda')
                self.gpu_model = self.gpu_model.to(self.gpu_device)
                self.gpu_batch_converter = alphabet.get_batch_converter()
                self.device = self.gpu_device
                print("✅ GPU模型加载成功")
        except Exception as e:
            print(f"❌ GPU模型加载失败: {e}")
        try:
            print("🖥️ 加载CPU模型作为备用...")
            self.cpu_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()  # 替换为 35M 模型
            self.cpu_device = torch.device('cpu')
            self.cpu_model = self.cpu_model.to(self.cpu_device)
            self.cpu_batch_converter = alphabet.get_batch_converter()
            if self.device is None:
                self.device = self.cpu_device
            print("✅ CPU模型加载成功")
        except Exception as e:
            print(f"❌ CPU模型加载失败: {e}")
            raise

    def _extract_batch_features(self, batch_data, use_gpu=True):
        try:
            model = self.gpu_model if use_gpu and self.gpu_model else self.cpu_model
            batch_converter = self.gpu_batch_converter if use_gpu and self.gpu_model else self.cpu_batch_converter
            device = self.gpu_device if use_gpu and self.gpu_model else self.cpu_device

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=False)  # 修改为第6层
                token_representations = results["representations"][6]  # 修改为第6层
            seq_lengths = (batch_tokens != model.alphabet.padding_idx).sum(1)
            batch_features = [token_representations[i, :seq_lengths[i]].mean(0).cpu().numpy() for i in
                              range(token_representations.size(0))]

            del batch_tokens, results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return batch_features
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_gpu:
                return self._extract_batch_features(batch_data, use_gpu=False)
            raise

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        if cache_path and os.path.exists(cache_path):
            print(f"📂 从缓存加载特征: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        features = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
            features.extend(self._extract_batch_features(batch_data))

            if (i // batch_size) % 10 == 0:
                print(f"📊 进度: {min(i + batch_size, len(sequences))}/{len(sequences)}")

        features_array = np.array(features)
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(features_array, f)
        return features_array


# ==========================================
# 缓存函数 - 正确使用st.cache_resource
# ==========================================
@st.cache_resource
def get_feature_extractor():
    """获取特征提取器的缓存实例"""
    return ESMFeatureExtractor()


@st.cache_resource
def load_model_and_scaler():
    """加载预训练模型和标准化器"""
    with st.spinner("🔄 正在加载预训练模型..."):
        # 检查模型文件是否存在
        model_path = "best_mutual_learning_model.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件未找到: {model_path}")
            st.info("请确保模型文件与应用在同一目录下")
            return None, None, None

        # 加载模型 - 修复安全警告
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # 尝试使用 weights_only=True (PyTorch 2.1+)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            checkpoint = torch.load(model_path, map_location=device)

        # 初始化模型
        model = MutualLearningModel(input_dim=480, num_classes=2).to(device)  # 修改 input_dim
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 获取标准化器
        scaler = checkpoint['scaler']
        return model, scaler, device


# ==========================================
# 应用主函数
# ==========================================
def main():
    # 页面配置
    st.set_page_config(
        page_title="Porcine Intestinal Virus Prediction System",
        page_icon="🐷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 应用标题和说明
    st.title("🦠 Porcine Intestinal Virus Protein Sequence Binary Classification System")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3>🔬 System Description</h3>
    <p>This system uses the DynML_Net model to classify protein sequences and determine if they are porcine intestinal viruses.</p>
    <ul>
    <li><b>Label-0</b>: Porcine Intestinal Virus (PEV)</li>
    <li><b>Label-1</b>: Non-Porcine Intestinal Virus (non-PEV)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ System Settings")
        st.markdown("### Model Information")
        st.info("Feature Extraction Method\n:ESM-2")
        st.info("Model Used\n:DynML_Net")
        st.markdown("### Usage Instructions")
        st.markdown("""
        1. **Single Sequence Prediction**: Paste the protein sequence in the input box
        2. **Batch Prediction**: Upload a CSV file containing sequences
        3. View prediction results and confidence scores
        """)
        st.markdown("### Notes")
        st.warning("""
        - Supports standard amino acid characters
        - GPU acceleration can significantly improve processing speed
        """)
        st.warning("""
        - Demo Purpose: This system is currently deployed solely for functional demonstration purposes.
        - Memory Constraints: The server is equipped with only 1GB of RAM. Since the ESM-2 650M model requires over 3GB of memory, this demo utilizes the lighter ESM-3 35M version due to insufficient resources.
        - Hardware Limitations & Local Deployment: As the model runs on the server's CPU without GPU acceleration, runtime errors may occur. This system is designed to showcase functionality only. For full-performance prediction capabilities, please deploy the system locally. Refer to our [GitHub Repository] for local deployment instructions.
        - Future Plans: Subject to sufficient funding, we plan to lease high-performance servers to support robust online inference in the future.
        """)

    # 加载模型和特征提取器
    model, scaler, device = load_model_and_scaler()
    feature_extractor = get_feature_extractor()

    if model is None or feature_extractor is None:
        st.stop()

    # 预测函数
    def predict_sequences(sequences: List[str]) -> List[dict]:
        """对序列列表进行预测"""
        if not sequences:
            return []

        # 提取特征
        with st.spinner(f"🧬 Extracting features for {len(sequences)} sequences..."):
            features = feature_extractor.extract_features(sequences)

        # 标准化
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(device)

        # 预测
        results = []
        with torch.no_grad():
            _, _, _, o_fused = model(features_tensor)
            probs = F.softmax(o_fused, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = probs[:, 1].cpu().numpy()  # 非猪肠道病毒的概率

        # 生成结果
        for i, (seq, pred, conf) in enumerate(zip(sequences, preds, confidences)):
            result = {
                'sequence_id': f"seq_{i + 1}",
                'sequence': seq[:50] + "..." if len(seq) > 50 else seq,
                'full_sequence': seq,
                'prediction': int(pred),
                'confidence': float(conf),
                'class_name': "non-PE" if pred == 1 else "PEV"
            }
            results.append(result)
        return results

    # 主界面 - 两种输入方式
    input_option = st.radio("Select Input Method", ["Single Sequence Prediction", "Batch CSV Prediction"], horizontal=True)

    if input_option == "Single Sequence Prediction":
        st.subheader("🔤  Enter Protein Sequence")
        sequence_input = st.text_area(
            "Paste protein sequence (supports standard amino acid characters)",
            height=150,
            placeholder="e.g., MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDMKHLKKADLIICAPNSYKKDDKPNQIKLLAVPTVMTKDDKQLLQEINELQDVVQDLRSLVEKNQIPAVDRAVTLTQRGELQAAGDKTLQEAVDRLQDKLQSLAEEGVKALQEELRKQLEAVDRAVTKLEQKLQDQVEALQARVDSLQAELRALQAQLAELQAELQALRSQLDELQAQLAELQAQLQELQAQLSELQSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQ"
        )

        if st.button("🔍 Start Prediction", type="primary"):
            if not sequence_input.strip():
                st.warning("⚠️ Please enter a valid protein sequence")
            else:
                # 预处理序列 - 只保留标准氨基酸
                sequence = ''.join(filter(str.isalpha, sequence_input.strip().upper()))
                sequence = ''.join([aa for aa in sequence if aa in 'ACDEFGHIKLMNPQRSTVWYBX'])
                if len(sequence) < 10:
                    st.error("❌ Sequence too short, please enter a sequence with at least 10 amino acids")
                elif len(sequence) > 10000:
                    st.error("❌ Sequence too long, maximum supported length is 10,000 amino acid")
                else:
                    # 进行预测
                    results = predict_sequences([sequence])

                    # 显示结果
                    st.subheader("📊 Prediction Results")
                    result = results[0]

                    # 使用卡片式布局展示结果
                    if result['prediction'] == 0:
                        color = "#ff4b4b"  # 红色表示猪肠道病毒
                        emoji = "🐷"
                    else:
                        color = "#1f77b4"  # 蓝色表示非猪肠道病毒
                        emoji = "🦠"

                    st.markdown(f"""
                    <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 15px; border-radius: 0 8px 8px 0; margin: 15px 0;">
                    <h3 style="color: {color};">{emoji} 预测结果: {result['class_name']}</h3>
                    <p><b>置信度:</b> {result['confidence']:.2%}</p>
                    <p><b>序列预览:</b> {result['sequence']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 显示详细置信度
                    st.subheader("📈 Confidence Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PEV", f"{1 - result['confidence']:.2%}")
                    with col2:
                        st.metric("non-PEV", f"{result['confidence']:.2%}")

                    # 可视化置信度
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 2))
                    classes = ['PEV', 'non-PEV']
                    probabilities = [1 - result['confidence'], result['confidence']]
                    colors = ['#ff4b4b', '#1f77b4']
                    bars = ax.barh(classes, probabilities, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_title('Forecast probability distribution')
                    ax.bar_label(bars, fmt='%.2f', padding=3)
                    st.pyplot(fig)

                    # 显示完整序列
                    with st.expander("📋 View Full Sequence"):
                        st.code(result['full_sequence'])

    else:  # 批量CSV预测
        st.subheader("📁 Upload CSV File")
        st.markdown("""
        Please upload a CSV file containing protein sequences. The file must include a `Sequence` column.
        
        **Example Format:**
        ```
        ID,Sequence
        seq1,MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHF
        seq2,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGV
        ```
        """)
        uploaded_file = st.file_uploader("Choose CSV Fil", type=["csv"])

        if uploaded_file is not None:
            try:
                # 读取CSV
                df = pd.read_csv(uploaded_file)
                if 'Sequence' not in df.columns:
                    st.error("❌ 'Sequence' column missing in CSV file")
                else:
                    st.success(f"✅ Successfully loaded {len(df)} sequences")

                    # 预览数据
                    with st.expander("🔍 Data Preview"):
                        st.dataframe(df.head())

                    # 预处理序列
                    sequences = []
                    valid_indices = []
                    for idx, row in df.iterrows():
                        seq = str(row['Sequence']).strip().upper()
                        # 仅保留标准氨基酸
                        seq_clean = ''.join([aa for aa in seq if aa in 'ACDEFGHIKLMNPQRSTVWYBX'])
                        if len(seq_clean) >= 10 and len(seq_clean) <= 10000:
                            sequences.append(seq_clean)
                            valid_indices.append(idx)

                    st.info(f"ℹ️ Valid sequences: {len(sequences)}/{len(df)} (Filtered out sequences that were too short, too long, or contained invalid characters)")

                    if st.button("🚀 Start Batch Prediction", type="primary"):
                        if not sequences:
                            st.warning("⚠️ No valid sequences to predic")
                        else:
                            # 进行预测
                            with st.spinner(f"🧠 Predicting {len(sequences)} sequences..."):
                                start_time = time.time()
                                results = predict_sequences(sequences)
                                elapsed_time = time.time() - start_time

                            # 创建结果DataFrame
                            results_df = pd.DataFrame(results)
                            results_df = results_df[['sequence_id', 'sequence', 'prediction', 'confidence', 'class_name']]

                            # 与原始数据合并
                            result_indices = pd.Series(valid_indices, name='original_index')
                            results_with_index = pd.concat([result_indices, results_df], axis=1)

                            # 创建完整的输出
                            output_df = df.copy()
                            output_df['Prediction'] = "无效序列"
                            output_df['Class'] = "无效序列"
                            output_df['Confidence'] = 0.0
                            for _, row in results_with_index.iterrows():
                                idx = int(row['original_index'])
                                output_df.at[idx, 'Prediction'] = row['prediction']
                                output_df.at[idx, 'Class'] = row['class_name']
                                output_df.at[idx, 'Confidence'] = row['confidence']

                            # 显示统计信息
                            st.subheader("📊 Prediction Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_valid = len(sequences)
                                st.metric("Valid Sequences", total_valid)
                            with col2:
                                pig_virus_count = sum(1 for r in results if r['prediction'] == 0)
                                st.metric("PEV", pig_virus_count)
                            with col3:
                                non_pig_count = total_valid - pig_virus_count
                                st.metric("non-PEV", non_pig_count)

                            st.success(f"✅  Prediction complete! Time elapsed: {elapsed_time:.2f} seconds")

                            # 显示结果预览
                            st.subheader("🔍 Results Preview")
                            st.dataframe(output_df.head(10))

                            # 下载结果
                            csv = output_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv",
                                type="primary"
                            )

                            # 可视化
                            st.subheader("📈 Result Distribution")
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 6))
                            class_counts = output_df[output_df['Prediction'] != "无效序列"]['Class'].value_counts()
                            colors = ['#ff4b4b', '#1f77b4']
                            bars = class_counts.plot(kind='bar', color=colors, ax=ax)
                            ax.set_title('Forecast category distribution', fontsize=16)
                            ax.set_xlabel('Category', fontsize=12)
                            ax.set_ylabel('Quantity', fontsize=12)
                            ax.tick_params(axis='x', rotation=0)
                            # 添加数据标签
                            for i, v in enumerate(class_counts.values):
                                ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ 处理文件时出错: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
