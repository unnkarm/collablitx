"""
Analytics Module
Advanced visualizations and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from ast import literal_eval
import json
from collections import Counter


st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


DATA_PATH = "indian_students (1).csv"
REVIEWS_PATH = "peer_reviews.json"

@st.cache_data(show_spinner=False)
def load_dataset(path):
    """Load student dataset"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name","year","skills","interests","bio","created_at"])

    df = pd.read_csv(path)

    for col in ["skills", "interests"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[")
                else (x.split(",") if isinstance(x, str) else [])
            )
        else:
            df[col] = [[] for _ in range(len(df))]

    return df

def load_reviews():
    """Load reviews"""
    if os.path.exists(REVIEWS_PATH):
        with open(REVIEWS_PATH, 'r') as f:
            return json.load(f)
    return []


if 'df_students' not in st.session_state:
    st.session_state.df_students = load_dataset(DATA_PATH)

if 'reviews' not in st.session_state:
    st.session_state.reviews = load_reviews()

df = st.session_state.df_students
reviews = st.session_state.reviews


st.markdown("""
<div class="main-header">
    <h1>📊 Analytics Dashboard</h1>
    <p>Comprehensive insights powered by data analysis and AI</p>
</div>
""", unsafe_allow_html=True)


if len(df) == 0:
    st.warning("⚠️ No data available. Add students first!")
    if st.button("➕ Go to Student Management"):
        st.switch_page("pages/1_👥_Student_Management.py")
    st.stop()


tabs = st.tabs(["📈 Overview", "👥 Student Analytics", "⭐ Review Analytics", "🎯 Skill Analysis", "📊 Advanced Insights"])

###########################################
# Tab 1: Overview
###########################################
with tabs[0]:
    st.subheader("📈 Platform Overview")
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">{}</div>'.format(len(df)), unsafe_allow_html=True)
        st.markdown('<p>Total Students</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_skills = sum(len(skills) for skills in df['skills'])
        st.markdown('<div class="stat-value">{}</div>'.format(total_skills), unsafe_allow_html=True)
        st.markdown('<p>Total Skills</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_skills = len(set(s for skills in df['skills'] for s in skills))
        st.markdown('<div class="stat-value">{}</div>'.format(unique_skills), unsafe_allow_html=True)
        st.markdown('<p>Unique Skills</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">{}</div>'.format(len(reviews)), unsafe_allow_html=True)
        st.markdown('<p>Total Reviews</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Student Distribution by Year")
        year_counts = df["year"].value_counts().reset_index()
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
        fig.update_layout(
            showlegend=False, 
            height=400,
            xaxis_title="Year",
            yaxis_title="Number of Students"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💼 Top 10 Skills")
        skills = [s for row in df["skills"] for s in row]
        if skills:
            freq = pd.Series(skills).value_counts().head(10).reset_index()
            freq.columns = ["skill", "count"]
            
            fig = px.bar(
                freq, 
                x="count", 
                y="skill",
                orientation='h',
                color="count",
                color_continuous_scale="Viridis",
                text="count"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False, 
                height=400,
                xaxis_title="Count",
                yaxis_title="Skill"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 Year Distribution (Pie)")
        fig = px.pie(
            year_counts, 
            names="year", 
            values="count",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top 8 Interests")
        interests = [i for row in df["interests"] for i in row]
        if interests:
            int_freq = pd.Series(interests).value_counts().head(8).reset_index()
            int_freq.columns = ["interest", "count"]
            
            fig = px.pie(
                int_freq, 
                names="interest", 
                values="count",
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    most_common_year = year_counts.iloc[0]['year']
    most_common_skill = pd.Series(skills).value_counts().index[0] if skills else "N/A"
    avg_skills = df['skills'].apply(len).mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h3>📚 Most Common Year</h3>
            <p style="font-size: 2rem; font-weight: bold;">Year {most_common_year}</p>
            <p>{year_counts.iloc[0]['count']} students</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>🔥 Hottest Skill</h3>
            <p style="font-size: 1.5rem; font-weight: bold;">{most_common_skill}</p>
            <p>{pd.Series(skills).value_counts().iloc[0] if skills else 0} students have this</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>📊 Avg Skills per Student</h3>
            <p style="font-size: 2rem; font-weight: bold;">{avg_skills:.1f}</p>
            <p>Skills per person</p>
        </div>
        """, unsafe_allow_html=True)

###########################################
# Tab 2: Student Analytics
###########################################
with tabs[1]:
    st.subheader("👥 Student Analytics")
    
    # Skills per student distribution
    st.markdown("### 📊 Skills Distribution")
    skills_per_student = df['skills'].apply(len)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            skills_per_student, 
            nbins=20,
            color_discrete_sequence=['#667eea'],
            labels={'value': 'Number of Skills', 'count': 'Number of Students'}
        )
        fig.update_layout(
            title="Distribution of Skills per Student",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            y=skills_per_student,
            color_discrete_sequence=['#764ba2'],
            labels={'y': 'Number of Skills'}
        )
        fig.update_layout(
            title="Skills per Student (Box Plot)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("### 🏆 Top Students by Skill Count")
    
    df_with_counts = df.copy()
    df_with_counts['skill_count'] = df['skills'].apply(len)
    df_with_counts['interest_count'] = df['interests'].apply(len)
    top_students = df_with_counts.nlargest(10, 'skill_count')
    
    fig = px.bar(
        top_students,
        x='name',
        y='skill_count',
        color='year',
        color_discrete_sequence=px.colors.qualitative.Set2,
        text='skill_count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Student Name",
        yaxis_title="Number of Skills",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("### 📈 Skills vs Interests Correlation")
    
    fig = px.scatter(
        df_with_counts,
        x='skill_count',
        y='interest_count',
        size='skill_count',
        color='year',
        hover_data=['name'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={'skill_count': 'Number of Skills', 'interest_count': 'Number of Interests'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("### 📋 Student Profiles Summary")
    
    summary_df = df_with_counts[['name', 'year', 'skill_count', 'interest_count']].copy()
    summary_df.columns = ['Name', 'Year', 'Skills', 'Interests']
    summary_df = summary_df.sort_values('Skills', ascending=False)
    
    st.dataframe(
        summary_df.style.background_gradient(subset=['Skills', 'Interests'], cmap='Purples'),
        use_container_width=True,
        height=400
    )

###########################################
# Tab 3: Review Analytics
###########################################
with tabs[2]:
    st.subheader("⭐ Review Analytics")
    
    if len(reviews) == 0:
        st.info("📊 No reviews yet. Submit reviews to see analytics!")
    else:
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-value">{avg_rating:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<p>Average Rating</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            highest = max(reviews, key=lambda x: x['rating'])
            st.markdown(f'<div class="stat-value">{highest["rating"]}⭐</div>', unsafe_allow_html=True)
            st.markdown(f'<p>{highest["reviewee"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            reviewers = len(set(r['reviewer'] for r in reviews))
            st.markdown(f'<div class="stat-value">{reviewers}</div>', unsafe_allow_html=True)
            st.markdown('<p>Unique Reviewers</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            reviewees = len(set(r['reviewee'] for r in reviews))
            st.markdown(f'<div class="stat-value">{reviewees}</div>', unsafe_allow_html=True)
            st.markdown('<p>Reviewed Students</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
           
            st.markdown("### ⭐ Rating Distribution")
            rating_counts = pd.Series([r['rating'] for r in reviews]).value_counts().sort_index()
            
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                color=rating_counts.values,
                color_continuous_scale='RdYlGn',
                labels={'x': 'Rating', 'y': 'Count'},
                text=rating_counts.values
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            
            st.markdown("### 🎯 Overall Performance")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_rating,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Rating"},
                delta={'reference': 4.0},
                gauge={
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 3.5], 'color': "gray"},
                        {'range': [3.5, 5], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4.5
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Skill ratings analysis
        st.markdown("### 🎯 Skill Ratings Analysis")
        
        skill_ratings = {'communication': [], 'technical': [], 'teamwork': [], 'problemSolving': []}
        for review in reviews:
            for skill, rating in review.get('skills', {}).items():
                if skill in skill_ratings:
                    skill_ratings[skill].append(rating)
        
        skill_avgs = {k.replace('_', ' ').title(): np.mean(v) if v else 0 
                     for k, v in skill_ratings.items()}
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(skill_avgs.values()),
            theta=list(skill_avgs.keys()),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### 🏆 Most Reviewed Students")
        
        reviewee_counts = Counter(r['reviewee'] for r in reviews)
        top_reviewees = pd.DataFrame(reviewee_counts.most_common(10), columns=['Student', 'Reviews'])
        
        fig = px.bar(
            top_reviewees,
            x='Reviews',
            y='Student',
            orientation='h',
            color='Reviews',
            color_continuous_scale='Blues',
            text='Reviews'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

###########################################
# Tab 4: Skill Analysis
###########################################
with tabs[3]:
    st.subheader("🎯 Deep Skill Analysis")
    
    all_skills = [s for skills in df['skills'] for s in skills]
    
    if not all_skills:
        st.info("No skills data available.")
    else:
        
        st.markdown("### 📊 Complete Skill Frequency")
        
        skill_counts = pd.Series(all_skills).value_counts().reset_index()
        skill_counts.columns = ['Skill', 'Count']
        
        fig = px.treemap(
            skill_counts.head(20),
            path=['Skill'],
            values='Count',
            color='Count',
            color_continuous_scale='Purples'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### 🏷️ Skill Categories")
        
        categories = {
            'Programming': ['Python', 'Java', 'C++', 'JavaScript', 'C', 'Go', 'Rust'],
            'Web Development': ['React', 'Node.js', 'HTML', 'CSS', 'Django', 'Express.js', 'Vue.js'],
            'Data Science': ['Machine Learning', 'Data Science', 'Pandas', 'NumPy', 'TensorFlow'],
            'Mobile': ['Flutter', 'React Native', 'Android', 'iOS', 'Swift', 'Kotlin'],
            'Cloud/DevOps': ['AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Azure', 'GCP'],
            'Design': ['UI/UX', 'Figma', 'Photoshop', 'Illustrator', 'Design'],
            'Database': ['SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'DB']
        }
        
        category_counts = {}
        for category, skills in categories.items():
            count = sum(1 for s in all_skills if any(skill.lower() in s.lower() for skill in skills))
            category_counts[category] = count
        
        cat_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        cat_df = cat_df[cat_df['Count'] > 0].sort_values('Count', ascending=False)
        
        fig = px.pie(
            cat_df,
            names='Category',
            values='Count',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### 🔗 Common Skill Combinations")
        
        top_skills = skill_counts.head(10)['Skill'].tolist()
        
        
        cooccurrence = pd.DataFrame(0, index=top_skills, columns=top_skills)
        
        for skills in df['skills']:
            for i, skill1 in enumerate(top_skills):
                for skill2 in top_skills[i+1:]:
                    if skill1 in skills and skill2 in skills:
                        cooccurrence.loc[skill1, skill2] += 1
                        cooccurrence.loc[skill2, skill1] += 1
        
        fig = px.imshow(
            cooccurrence,
            color_continuous_scale='Purples',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

###########################################
# Tab 5: Advanced Insights
###########################################
with tabs[4]:
    st.subheader("📊 Advanced Insights")
    
    
    st.markdown("### 📈 Growth Trends")
    
    if 'created_at' in df.columns:
        df_dates = df.copy()
        df_dates['created_at'] = pd.to_datetime(df_dates['created_at'], errors='coerce')
        df_dates = df_dates.dropna(subset=['created_at'])
        
        if len(df_dates) > 0:
            df_dates['date'] = df_dates['created_at'].dt.date
            growth = df_dates.groupby('date').size().cumsum().reset_index()
            growth.columns = ['Date', 'Cumulative Students']
            
            fig = px.line(
                growth,
                x='Date',
                y='Cumulative Students',
                markers=True,
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("### 🌈 Diversity Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_diversity = len(df['year'].unique())
        st.metric("Year Diversity", f"{year_diversity} different years")
    
    with col2:
        skill_diversity = len(set(all_skills))
        st.metric("Skill Diversity", f"{skill_diversity} unique skills")
    
    with col3:
        all_interests = [i for interests in df['interests'] for i in interests]
        interest_diversity = len(set(all_interests))
        st.metric("Interest Diversity", f"{interest_diversity} unique interests")
    
   
    st.markdown("---")
    st.markdown("### 💡 AI-Powered Recommendations")
    
    recommendations = []
    
   
    if skill_diversity < 20:
        recommendations.append("📚 Consider adding more diverse skills to expand capabilities")
    
    
    if len(reviews) < len(df):
        recommendations.append("⭐ Encourage more peer reviews to build collaboration culture")
    
    
    max_year_pct = (df['year'].value_counts().iloc[0] / len(df)) * 100
    if max_year_pct > 60:
        recommendations.append("👥 Try to balance student distribution across different years")
    
    if not recommendations:
        recommendations.append("✅ Everything looks great! Keep up the good work!")
    
    for rec in recommendations:
        st.info(rec)


st.markdown("---")
if st.button("🏠 Back to Home"):
    st.switch_page("Dashboard.py")
    