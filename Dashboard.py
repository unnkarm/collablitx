import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="AI Student Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    /* MODIFIED: Removed dynamic effects (cursor, transition, hover) for static cards */
    .nav-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid transparent;
        height: 100%;
    }
    /* REMOVED: .nav-card:hover styles */
    .nav-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .nav-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .nav-desc {
        color: #666;
        font-size: 0.9rem;
    }
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# Removed the session state logic as it is no longer needed for manual page switching
# if 'current_page' not in st.session_state:
#     st.session_state.current_page = 'home'


st.markdown("""
<div class="main-header">
    <h1>🎓 AI-Powered Student Collaboration Platform</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Your complete solution for student networking, project collaboration, and peer evaluation
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">4</div>
        <div class="stat-label">Modules</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="stat-value">AI</div>
        <div class="stat-label">Powered</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="stat-value">∞</div>
        <div class="stat-label">Students</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="stat-value">⭐</div>
        <div class="stat-label">Reviews</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("## 🚀 Choose a Module")
st.info("Since this is a multi-page app, please use the **Sidebar Navigation** on the left to access the modules.")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">👥</div>
        <div class="nav-title">Student Management</div>
        <div class="nav-desc">Manage student database and add new members</div>
        <br>
        <span class="feature-badge">Database</span>
        <span class="feature-badge">Add Students</span>
        <span class="feature-badge">CSV Import</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons and switch_page removed for static navigation
    # if st.button("🚀 Launch Student Management", use_container_width=True, key="btn1"):
    #     st.switch_page("Student Management.py")


with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🔍</div>
        <div class="nav-title">Find Teammate</div>
        <div class="nav-desc">AI-powered teammate discovery and team building</div>
        <br>
        <span class="feature-badge">AI Matching</span>
        <span class="feature-badge">Team Builder</span>
        <span class="feature-badge">Smart Search</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons and switch_page removed for static navigation
    # if st.button("🚀 Launch Find Teammate", use_container_width=True, key="btn2"):
    #     st.switch_page("Find Teammate.py")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Analytics Dashboard</div>
        <div class="nav-desc">Visualize data and gain insights with AI analytics</div>
        <br>
        <span class="feature-badge">Visualizations</span>
        <span class="feature-badge">Insights</span>
        <span class="feature-badge">Reports</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons and switch_page removed for static navigation
    # if st.button("🚀 Launch Analytics", use_container_width=True, key="btn3"):
    #     st.switch_page("Analytics.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">⭐</div>
        <div class="nav-title">Peer Reviews</div>
        <div class="nav-desc">Submit and manage peer evaluations with sentiment analysis</div>
        <br>
        <span class="feature-badge">Reviews</span>
        <span class="feature-badge">Ratings</span>
        <span class="feature-badge">Sentiment AI</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons and switch_page removed for static navigation
    # if st.button("🚀 Launch Peer Reviews", use_container_width=True, key="btn4"):
    #     st.switch_page("Peer Review.py")


st.markdown("---")
st.markdown("## ✨ Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🤖 AI-Powered Matching
    - Smart recommendation engine
    - Skill-based similarity
    - NLP text understanding
    - Diversity-aware algorithms
    """)

with col2:
    st.markdown("""
    ### 📈 Advanced Analytics
    - Real-time visualizations
    - Performance metrics
    - Sentiment analysis
    - Trend tracking
    """)

with col3:
    st.markdown("""
    ### 🔒 Data Management
    - Secure storage
    - CSV import/export
    - Backup systems
    - Version control
    """)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🎓 AI-Powered Student Collaboration Platform</strong></p>
    <p>Built with Streamlit • Powered by Hugging Face • Enhanced with Machine Learning</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        💡 <strong>Quick Start:</strong> Click the page links in the **Sidebar** to navigate!
    </p>
</div>
""", unsafe_allow_html=True)