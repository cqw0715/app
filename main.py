import streamlit as st
import importlib.util
import sys
import os

# 保存原始的st.set_page_config函数
original_set_page_config = st.set_page_config

# 自定义的set_page_config，不做任何操作
def dummy_set_page_config(*args, **kwargs):
    pass

# 设置主页面配置
original_set_page_config(
    page_title="猪肠道病毒预测系统",
    page_icon="🦠",
    layout="wide"
)

# 应用标题
st.title("🐷 Porcine Enterovirus Prediction System V1.0 (Demo Version)")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <h3>🔬 System Introduction</h3>
    <p>This platform integrates two types of prediction and identification tasks:</p>
    <ul>
        <li><b>Porcine Intestinal Virus Binary Classification Model</b>: Identifies whether a protein sequence is a porcine intestinal virus</li>
        <li><b>Porcine Intestinal Virus Multi-classification Model</b>: Identifies 8 different common types of porcine intestinal viruses</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <p><center>
    Please select a tab to switch to different porcine intestinal virus prediction task modules
    </center></p>
</div>
</div>
""", unsafe_allow_html=True)

# 创建标签页
tab1, tab2 = st.tabs(["🦠  Porcine Intestinal Virus Binary Classification Model", "🦠 Porcine Intestinal Virus Multi-classification Model"])

# 加载并运行模型的函数
def run_model(tab, model_file, model_name):
    with tab:
        # 临时替换st.set_page_config
        st.set_page_config = dummy_set_page_config
        
        try:
            # 检查文件是否存在
            if not os.path.exists(model_file):
                st.error(f"❌ 模型文件未找到: {model_file}")
                st.info("请确保模型文件与本应用在同一目录下")
                return
                
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[model_name] = module
            spec.loader.exec_module(module)
            
            # 运行模型的main函数
            if hasattr(module, 'main'):
                module.main()
            else:
                st.error(f"❌ 模型 {model_name} 没有定义 main() 函数")
        except Exception as e:
            st.error(f"❌ 加载模型 {model_name} 时出错: {str(e)}")
            st.exception(e)
        finally:
            # 恢复原始的st.set_page_config
            st.set_page_config = original_set_page_config

# 在各自的标签页中运行模型
run_model(tab1, "1.py", "model_pev")
run_model(tab2, "2.py", "model_multiclass")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>Porcine Enterovirus Prediction and Identification System (Demo Version) &copy; 2026 | School of Artificial Intelligence, Anhui Agricultural University</p>
</div>
""", unsafe_allow_html=True)
