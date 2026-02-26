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
st.title("🦠 猪肠道病毒预测系统V1.O（功能展示版）")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <h3>🔬 系统简介</h3>
    <p>本平台整合了两种病毒预测模型:</p>
    <ul>
        <li><b>猪肠道病毒二分类模型</b>: 识别序列是否为猪肠道病毒</li>
        <li><b>猪肠道病毒多分类模型</b>: 识别8种不同病毒类型</li>
    </ul>
    <p><center>请选择标签页切换不同预测系统</center></p>
</div>
""", unsafe_allow_html=True)

# 创建标签页
tab1, tab2 = st.tabs(["🐷 猪肠道病毒二分类模型", "🦠 猪肠道病毒多分类模型"])

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

st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <p><center>
    <b>注</b>：由于模型运行在服务器CPU上，会导致运行错误，本系统仅展示相关系统功能，<br>
    并且该服务器只有1G的内存，而ESM-2 650M需要3G以上的内存，由于内存不足，该系统使用的是ESM-3 35M版本；<br>
    但满血版预测系统可在本地进行部署，请参考（github）进行本地部署</center></p>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <p><center>
    经费充足时会考虑租借更好的服务器以满足在线识别的需求
    </center></p>
</div>
</div>
""", unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>猪肠道病毒预测系统V1.0 &copy; 2026 | 安徽农业大学 人工智能学院</p>
</div>
""", unsafe_allow_html=True)
