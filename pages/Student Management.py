"""
Student Management Module
Manage student database and add new students
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from ast import literal_eval
from datetime import datetime


st.set_page_config(
    page_title="Student Management",
    page_icon="👥",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .skill-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .student-card {
        background: skyblue;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .student-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


DATA_PATH = "indian_students (1).csv"
BACKUP_PATH = "indian_students_backup.csv"

@st.cache_data(show_spinner=False)
def load_dataset(path):
    """Load student dataset from CSV"""
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["name","year","skills","interests","bio","created_at"])
        df.to_csv(path, index=False)
        return df

    df = pd.read_csv(path)

    for col in ["skills", "interests"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[")
                else (x.split(",") if isinstance(x, str) else [])
            )
        else:
            df[col] = [[] for _ in range(len(df))]

    if "created_at" not in df.columns:
        df["created_at"] = pd.NaT

    return df

def save_dataset(df, path):
    """Save dataset with backup"""
    if os.path.exists(path):
        df.to_csv(BACKUP_PATH, index=False)
    df.to_csv(path, index=False)


if 'df_students' not in st.session_state:
    st.session_state.df_students = load_dataset(DATA_PATH)


st.markdown("""
<div class="main-header">
    <h1>👥 Student Management</h1>
    <p>Manage your student database and add new members to the platform</p>
</div>
""", unsafe_allow_html=True)


tabs = st.tabs(["📚 View Database", "➕ Add Student", "📊 Quick Stats"])

###########################################
# Tab 1: View Database
###########################################
with tabs[0]:
    st.subheader("📚 Student Database")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search students", placeholder="Search by name, skill, or interest...")
    
    with col2:
        year_filter = st.selectbox("Filter by Year", ["All"] + ["1", "2", "3", "4", "Other"])
    
    with col3:
        st.write("")
        st.write("")
        show_all = st.checkbox("Show all columns", value=False)
    
    
    filtered_df = st.session_state.df_students.copy()
    
    if search_query:
        mask = (
            filtered_df['name'].str.contains(search_query, case=False, na=False) |
            filtered_df['skills'].apply(lambda x: any(search_query.lower() in str(s).lower() for s in x)) |
            filtered_df['interests'].apply(lambda x: any(search_query.lower() in str(i).lower() for i in x))
        )
        filtered_df = filtered_df[mask]
    
    if year_filter != "All":
        filtered_df = filtered_df[filtered_df['year'] == year_filter]
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Total Students", len(st.session_state.df_students))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Filtered Results", len(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        total_skills = sum(len(skills) for skills in st.session_state.df_students['skills'])
        st.metric("Total Skills", total_skills)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        unique_skills = len(set(s for skills in st.session_state.df_students['skills'] for s in skills))
        st.metric("Unique Skills", unique_skills)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    display_option = st.radio("Display as:", ["Table View", "Card View"], horizontal=True)
    
    if display_option == "Table View":
        
        if show_all:
            st.dataframe(filtered_df, use_container_width=True, height=500)
        else:
            display_df = filtered_df[['name', 'year', 'skills', 'interests']].copy()
            st.dataframe(display_df, use_container_width=True, height=500)
        
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"students_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    else:
       
        if len(filtered_df) == 0:
            st.info("No students match your search criteria.")
        else:
            for idx, row in filtered_df.iterrows():
                st.markdown(f"""
                <div class="student-card">
                    <h3>{row['name']}</h3>
                    <p><strong>Year:</strong> {row['year']} | <strong>Added:</strong> {row.get('created_at', 'N/A')}</p>
                    <p><strong>Bio:</strong> {row.get('bio', 'No bio provided')}</p>
                    <p><strong>Skills:</strong><br>
                    {''.join([f'<span class="skill-badge">{s}</span>' for s in row['skills']])}
                    </p>
                    <p><strong>Interests:</strong><br>
                    {''.join([f'<span class="skill-badge" style="background: #fff3e0; color: #e65100;">{i}</span>' for i in row['interests']])}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                
                col1, col2, col3 = st.columns([4, 1, 1])
                with col3:
                    if st.button(f"🗑️ Delete", key=f"del_{idx}"):
                        st.session_state.df_students = st.session_state.df_students.drop(idx).reset_index(drop=True)
                        save_dataset(st.session_state.df_students, DATA_PATH)
                        st.success(f"Deleted {row['name']}")
                        st.rerun()
    
    
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("💾 Save & Backup Database", use_container_width=True):
            save_dataset(st.session_state.df_students, DATA_PATH)
            st.success("✅ Database saved and backed up!")

###########################################
# Tab 2: Add Student
###########################################
with tabs[1]:
    st.subheader("➕ Add New Student")
    
    st.markdown("""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        💡 <strong>Tip:</strong> Fill in all details to help AI make better recommendations!
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("add_student_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("👤 Full Name *", placeholder="e.g., John Doe")
            year = st.selectbox("📅 Academic Year *", ["1", "2", "3", "4", "Other"])
            skills = st.text_area(
                "💻 Skills (comma-separated) *", 
                placeholder="e.g., Python, React, Machine Learning, Data Analysis",
                height=100
            )
        
        with col2:
            interests = st.text_area(
                "🎯 Interests (comma-separated)", 
                placeholder="e.g., AI, Web Development, Robotics, Cloud Computing",
                height=100
            )
            bio = st.text_area(
                "📝 Short Bio", 
                placeholder="Tell us about yourself, your experience, and what you're passionate about...",
                height=100
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            submitted = st.form_submit_button("✨ Add Student", use_container_width=True)
        
        if submitted:
            if name and skills:
                new_row = {
                    "name": name.strip(),
                    "year": year,
                    "skills": [s.strip() for s in skills.split(",") if s.strip()],
                    "interests": [s.strip() for s in interests.split(",") if s.strip()],
                    "bio": bio.strip(),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.df_students = pd.concat(
                    [st.session_state.df_students, pd.DataFrame([new_row])], 
                    ignore_index=True
                )
                save_dataset(st.session_state.df_students, DATA_PATH)
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>🎉 Success!</h3>
                    <p><strong>{name}</strong> has been added to the database!</p>
                    <p>Skills added: {len(new_row['skills'])} | Interests: {len(new_row['interests'])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                
                
                with st.expander("👀 Preview Student Card"):
                    st.markdown(f"""
                    <div class="student-card">
                        <h3>{name}</h3>
                        <p><strong>Year:</strong> {year}</p>
                        <p><strong>Bio:</strong> {bio if bio else 'No bio provided'}</p>
                        <p><strong>Skills:</strong><br>
                        {''.join([f'<span class="skill-badge">{s}</span>' for s in new_row['skills']])}
                        </p>
                        <p><strong>Interests:</strong><br>
                        {''.join([f'<span class="skill-badge" style="background: #fff3e0; color: #e65100;">{i}</span>' for i in new_row['interests']])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Please fill in all required fields (marked with *)")

###########################################
# Tab 3: Quick Stats
###########################################
with tabs[2]:
    st.subheader("📊 Quick Statistics")
    
    if len(st.session_state.df_students) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.markdown("### 📅 Students by Year")
            year_counts = st.session_state.df_students["year"].value_counts().reset_index()
            year_counts.columns = ["year", "count"]
            fig = px.bar(
                year_counts, 
                x="year", 
                y="count",
                color="count",
                color_continuous_scale="Purples",
                text="count"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            
            st.markdown("### 💻 Top 10 Skills Distribution")
            skills = [s for row in st.session_state.df_students["skills"] for s in row]
            if skills:
                freq = pd.Series(skills).value_counts().head(10).reset_index()
                freq.columns = ["skill", "count"]
                fig = px.pie(
                    freq, 
                    names="skill", 
                    values="count",
                    color_discrete_sequence=px.colors.sequential.Purples_r
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        # Top interests
        st.markdown("### 🎯 Top Interests")
        interests = [i for row in st.session_state.df_students["interests"] for i in row]
        if interests:
            freq = pd.Series(interests).value_counts().head(15).reset_index()
            freq.columns = ["interest", "count"]
            fig = px.bar(
                freq, 
                x="count", 
                y="interest",
                orientation='h',
                color="count",
                color_continuous_scale="Viridis",
                text="count"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("---")
        st.markdown("### 🎓 Skill Coverage Analysis")
        
        all_skills = [s for row in st.session_state.df_students["skills"] for s in row]
        skill_freq = pd.Series(all_skills).value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Most Common Skill", skill_freq.index[0] if len(skill_freq) > 0 else "N/A", 
                     f"{skill_freq.iloc[0]} students" if len(skill_freq) > 0 else "")
        
        with col2:
            avg_skills = st.session_state.df_students['skills'].apply(len).mean()
            st.metric("Avg Skills per Student", f"{avg_skills:.1f}")
        
        with col3:
            students_with_interests = len([i for i in st.session_state.df_students['interests'] if len(i) > 0])
            pct = (students_with_interests / len(st.session_state.df_students)) * 100
            st.metric("Students with Interests", f"{pct:.0f}%")
        
    else:
        st.info("📊 No data available yet. Add some students to see statistics!")


st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=False):
    st.switch_page("Dashboard.py")