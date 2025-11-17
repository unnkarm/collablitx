"""
Find Teammates Module
AI-powered teammate discovery and team building
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from random import uniform
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="Find Teammates",
    page_icon="🔍",
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
    .teammate-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
        border-left: 4px solid #667eea;
    }
    .teammate-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .match-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
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
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


use_hf = True
embedder = None

try:
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import login
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token:
        try:
            login(token=hf_token)
        except:
            pass
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except:
    use_hf = False


DATA_PATH = "indian_students (1).csv"

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

def prepare_text(row):
    """Prepare text for embedding"""
    skills = row.get("skills", [])
    interests = row.get("interests", [])
    bio = row.get("bio", "")
    return f"{row.get('name','')} {' '.join(skills)} {' '.join(interests)} {bio}".strip()

@st.cache_data(show_spinner=False)
def compute_embeddings(_df):
    """Compute embeddings for students"""
    texts = _df.apply(prepare_text, axis=1).astype(str).tolist()

    if use_hf and embedder:
        return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(len(texts), 384))

def top_similar(idx, embeddings, k=5):
    """Find top similar students"""
    if len(embeddings) < 2:
        return []
    nn = NearestNeighbors(n_neighbors=min(k+1, len(embeddings)), metric="cosine").fit(embeddings)
    dist, ind = nn.kneighbors([embeddings[idx]])
    pairs = list(zip(ind[0], dist[0]))
    return [(i, d) for i, d in pairs if i != idx][:k]


if 'df_students' not in st.session_state:
    st.session_state.df_students = load_dataset(DATA_PATH)

df = st.session_state.df_students


st.markdown("""
<div class="main-header">
    <h1>🔍 Find Teammates</h1>
    <p>Discover the perfect collaborators using AI-powered matching algorithms</p>
</div>
""", unsafe_allow_html=True)

# AI Status
col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    if use_hf:
        st.markdown('<span class="ai-badge">✨ AI Enabled</span>', unsafe_allow_html=True)
    else:
        st.info("⚠️ AI disabled - using basic matching")


if len(df) == 0:
    st.warning("⚠️ No students in database. Add students first!")
    if st.button("➕ Go to Student Management"):
        st.switch_page("pages/1_👥_Student_Management.py")
    st.stop()


embeddings = compute_embeddings(df)


tabs = st.tabs(["🎯 Similar Students", "👥 AI Team Builder", "📋 Task Matcher", "🔍 Advanced Search"])

###########################################
# Tab 1: Similar Students
###########################################
with tabs[0]:
    st.subheader("🎯 Find Similar Students")
    
    st.markdown("""
    <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        💡 <strong>How it works:</strong> AI analyzes skills, interests, and bio to find the most compatible teammates!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        student_names = df['name'].tolist()
        selected_student = st.selectbox("Select a student to find matches", student_names)
    
    with col2:
        num_matches = st.slider("Number of matches", 3, 10, 5)
    
    if st.button("🔍 Find Similar Students", use_container_width=True):
        idx = student_names.index(selected_student)
        similar = top_similar(idx, embeddings, k=num_matches)
        
        
        st.markdown("### 👤 Selected Student")
        selected_row = df.loc[idx]
        
        st.markdown(f"""
        <div class="teammate-card">
            <h3>{selected_row['name']}</h3>
            <p><strong>Year:</strong> {selected_row['year']}</p>
            <p><strong>Bio:</strong> {selected_row.get('bio', 'No bio')}</p>
            <p><strong>Skills:</strong><br>
            {''.join([f'<span class="skill-badge">{s}</span>' for s in selected_row['skills'][:10]])}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.markdown("### 🎯 Top Matches")
        
        cols = st.columns(min(3, len(similar)))
        for i, (sim_idx, distance) in enumerate(similar):
            with cols[i % 3]:
                similarity = (1 - distance) * 100
                row = df.loc[sim_idx]
                
                
                medal = ""
                if i == 0:
                    medal = "🥇 "
                elif i == 1:
                    medal = "🥈 "
                elif i == 2:
                    medal = "🥉 "
                
                st.markdown(f"""
                <div class="teammate-card">
                    <h3>{medal}{row['name']}</h3>
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="match-badge">🎯 {similarity:.1f}% Match</span>
                    </div>
                    <p><strong>Year:</strong> {row['year']}</p>
                    <p><strong>Skills:</strong><br>
                    {''.join([f'<span class="skill-badge">{s}</span>' for s in row['skills'][:4]])}
                    {f'<span class="skill-badge">+{len(row["skills"]) - 4}</span>' if len(row['skills']) > 4 else ''}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        
        if len(similar) > 0:
            st.markdown("---")
            st.markdown("### 🕸️ Similarity Network")
            
            
            import plotly.graph_objects as go
            
            
            edge_x = []
            edge_y = []
            
            for sim_idx, distance in similar[:5]:
                edge_x.extend([0, (1-distance)*10, None])
                edge_y.extend([0, np.random.uniform(-5, 5), None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#667eea'),
                hoverinfo='none',
                mode='lines')
            
            node_x = [0] + [(1-d)*10 for _, d in similar[:5]]
            node_y = [0] + [np.random.uniform(-5, 5) for _ in similar[:5]]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    size=[30] + [20]*len(similar[:5]),
                    color=['#f5576c'] + ['#667eea']*len(similar[:5]),
                    line_width=2),
                text=[selected_student] + [df.loc[i, 'name'] for i, _ in similar[:5]],
                textposition="top center")
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=400
                          ))
            
            st.plotly_chart(fig, use_container_width=True)

###########################################
# Tab 2: AI Team Builder
###########################################
with tabs[1]:
    st.subheader("👥 AI Team Builder")
    
    st.markdown("""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        🤖 <strong>AI-Powered:</strong> Describe your project and let AI assemble the perfect team based on skills and expertise!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        project_desc = st.text_area(
            "📝 Project Description", 
            placeholder="Describe your project requirements, needed skills, and goals...\n\nExample: Building a mobile app for food delivery with React Native frontend, Node.js backend, and MongoDB database. Need expertise in UI/UX design and cloud deployment.",
            height=150
        )
    
    with col2:
        team_size = st.slider("👥 Team Size", 2, 10, 4)
        diversity = st.slider("🌈 Diversity Weight", 0.0, 1.0, 0.3, 0.1)
        
        st.markdown("""
        <small>
        <strong>Diversity Weight:</strong><br>
        Higher = more varied skills<br>
        Lower = more specialized
        </small>
        """, unsafe_allow_html=True)
    
    if st.button("✨ Generate Team", use_container_width=True):
        if project_desc:
            with st.spinner("🤖 AI is analyzing and forming your dream team..."):
                
                if use_hf and embedder:
                    req_emb = embedder.encode([project_desc], convert_to_numpy=True)[0]
                else:
                    req_emb = np.random.random(384)
                
                
                sims = cosine_similarity([req_emb], embeddings)[0]
                
                
                best_indices = []
                remaining = list(range(len(df)))
                
                for _ in range(min(team_size, len(df))):
                    if not remaining:
                        break
                    
                    scores = []
                    for idx in remaining:
                        base_score = sims[idx]
                        
                        
                        if best_indices:
                            diversity_scores = []
                            for prev_idx in best_indices:
                                sim_to_prev = cosine_similarity(
                                    [embeddings[idx]], 
                                    [embeddings[prev_idx]]
                                )[0][0]
                                diversity_scores.append(1 - sim_to_prev)
                            diversity_bonus = np.mean(diversity_scores) * diversity
                            final_score = base_score + diversity_bonus
                        else:
                            final_score = base_score
                        
                        scores.append((idx, final_score))
                    
                    best = max(scores, key=lambda x: x[1])
                    best_indices.append(best[0])
                    remaining.remove(best[0])
                
                
                st.markdown("### 🎉 Your AI-Generated Team")
                
                st.success(f"✅ Found {len(best_indices)} perfect teammates!")
                
                
                col1, col2, col3 = st.columns(3)
                all_team_skills = []
                for idx in best_indices:
                    all_team_skills.extend(df.loc[idx, 'skills'])
                
                with col1:
                    st.metric("Team Members", len(best_indices))
                with col2:
                    st.metric("Total Skills", len(all_team_skills))
                with col3:
                    st.metric("Unique Skills", len(set(all_team_skills)))
                
                st.markdown("---")
                
                
                cols = st.columns(min(3, len(best_indices)))
                for i, idx in enumerate(best_indices):
                    with cols[i % 3]:
                        row = df.loc[idx]
                        match_pct = sims[idx] * 100
                        
                        rank = ""
                        if i == 0:
                            rank = "🎖️ Team Lead"
                        elif i == 1:
                            rank = "⭐ Core Member"
                        else:
                            rank = "👤 Member"
                        
                        st.markdown(f"""
                        <div class="teammate-card">
                            <h3>{row['name']}</h3>
                            <p style="text-align: center; color: #666;">{rank}</p>
                            <div style="text-align: center; margin: 1rem 0;">
                                <span class="match-badge">Relevance: {match_pct:.1f}%</span>
                            </div>
                            <p><strong>Year:</strong> {row['year']}</p>
                            <p><strong>Skills:</strong><br>
                            {''.join([f'<span class="skill-badge">{s}</span>' for s in row['skills'][:3]])}
                            {f'<span class="skill-badge">+{len(row["skills"]) - 3}</span>' if len(row['skills']) > 3 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                
                st.markdown("---")
                st.markdown("### 🎯 Team Skill Coverage")
                
                skill_counts = pd.Series(all_team_skills).value_counts().head(15)
                fig = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    color=skill_counts.values,
                    color_continuous_scale='Purples',
                    labels={'x': 'Count', 'y': 'Skill'}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("⚠️ Please describe your project first!")

###########################################
# Tab 3: Task Matcher
###########################################
with tabs[2]:
    st.subheader("📋 Task Matcher")
    
    st.markdown("""
    <div style="background: #f3e5f5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        🎯 <strong>Smart Matching:</strong> Enter required skills and get personalized task recommendations!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        skills_input = st.text_input(
            "🎯 Required Skills (comma-separated)", 
            placeholder="e.g., Python, React, UI/UX, Machine Learning"
        )
    
    with col2:
        num_tasks = st.slider("Number of tasks", 1, 10, 5)
    
    
    curated_tasks = [
        {"task": "Build ML Pipeline", "tags": ["ML", "Python", "Pandas", "sklearn"], "difficulty": "Advanced"},
        {"task": "Frontend Dashboard", "tags": ["React", "JS", "UI", "CSS"], "difficulty": "Intermediate"},
        {"task": "Database Schema Design", "tags": ["SQL", "DB", "Backend", "PostgreSQL"], "difficulty": "Intermediate"},
        {"task": "Cloud Deployment", "tags": ["Docker", "CI/CD", "AWS", "DevOps"], "difficulty": "Advanced"},
        {"task": "Mobile App Development", "tags": ["Flutter", "React Native", "Mobile", "UI"], "difficulty": "Advanced"},
        {"task": "Data Visualization Dashboard", "tags": ["Python", "Plotly", "D3.js", "Data"], "difficulty": "Intermediate"},
        {"task": "REST API Development", "tags": ["Node.js", "Express", "API", "Backend"], "difficulty": "Intermediate"},
        {"task": "Machine Learning Model Training", "tags": ["Python", "TensorFlow", "ML", "AI"], "difficulty": "Advanced"},
        {"task": "UI/UX Design System", "tags": ["Figma", "Design", "UI/UX", "Prototyping"], "difficulty": "Intermediate"},
        {"task": "Automated Testing Suite", "tags": ["Python", "Jest", "Testing", "QA"], "difficulty": "Intermediate"},
        {"task": "Microservices Architecture", "tags": ["Docker", "Kubernetes", "Backend", "DevOps"], "difficulty": "Advanced"},
        {"task": "Data Scraping Pipeline", "tags": ["Python", "BeautifulSoup", "Data", "Automation"], "difficulty": "Beginner"},
        {"task": "GraphQL API", "tags": ["GraphQL", "Node.js", "API", "Backend"], "difficulty": "Intermediate"},
        {"task": "Progressive Web App", "tags": ["React", "PWA", "Web", "JavaScript"], "difficulty": "Intermediate"},
        {"task": "Blockchain Smart Contract", "tags": ["Solidity", "Blockchain", "Web3", "Ethereum"], "difficulty": "Advanced"},
    ]
    
    if st.button("🔍 Find Matching Tasks", use_container_width=True):
        if skills_input:
            skills_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
            scored = []
            
            for task in curated_tasks:
                overlap = len(set(t.lower() for t in task["tags"]) & set(skills_list))
                score = overlap + uniform(0, 0.5)  # Add small random for tie-breaking
                scored.append((task, score, overlap))
            
            
            scored = sorted(scored, key=lambda x: -x[1])[:num_tasks]
            
            st.markdown("### 📋 Recommended Tasks")
            
            for task, score, overlap in scored:
                match_pct = (overlap / len(skills_list)) * 100 if skills_list else 0
                
                difficulty_color = {
                    "Beginner": "#4caf50",
                    "Intermediate": "#ff9800",
                    "Advanced": "#f44336"
                }
                
                st.markdown(f"""
                <div class="teammate-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>📌 {task['task']}</h3>
                        <span class="match-badge">Match: {match_pct:.0f}%</span>
                    </div>
                    <p style="color: {difficulty_color[task['difficulty']]}; font-weight: bold;">
                        {task['difficulty']} Level
                    </p>
                    <p><strong>Required Skills:</strong><br>
                    {''.join([f'<span class="skill-badge" style="background: {"#c8e6c9" if t.lower() in skills_list else "#e3f2fd"};">{t}</span>' for t in task['tags']])}
                    </p>
                    <p><strong>Matching Skills:</strong> {overlap}/{len(skills_list)}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter required skills!")

###########################################
# Tab 4: Advanced Search
###########################################
with tabs[3]:
    st.subheader("🔍 Advanced Search & Filter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_name = st.text_input("👤 Search by name", placeholder="Enter name...")
    
    with col2:
        filter_year = st.multiselect("📅 Filter by year", ["1", "2", "3", "4", "Other"])
    
    with col3:
        filter_skills = st.multiselect("💻 Filter by skills", 
            list(set(s for row in df['skills'] for s in row))[:20])
    
    
    filtered_df = df.copy()
    
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
    
    if filter_year:
        filtered_df = filtered_df[filtered_df['year'].isin(filter_year)]
    
    if filter_skills:
        filtered_df = filtered_df[filtered_df['skills'].apply(
            lambda x: any(skill in x for skill in filter_skills)
        )]
    
    st.markdown(f"### 📊 Found {len(filtered_df)} students")
    
    if len(filtered_df) > 0:
        
        for idx, row in filtered_df.iterrows():
            st.markdown(f"""
            <div class="teammate-card">
                <h3>{row['name']}</h3>
                <p><strong>Year:</strong> {row['year']}</p>
                <p><strong>Bio:</strong> {row.get('bio', 'No bio')}</p>
                <p><strong>Skills:</strong><br>
                {''.join([f'<span class="skill-badge">{s}</span>' for s in row['skills'][:8]])}
                {f'<span class="skill-badge">+{len(row["skills"]) - 8}</span>' if len(row['skills']) > 8 else ''}
                </p>
                <p><strong>Interests:</strong><br>
                {''.join([f'<span class="skill-badge" style="background: #fff3e0; color: #e65100;">{i}</span>' for i in row['interests'][:5]])}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No students match your criteria. Try adjusting your filters.")


st.markdown("---")
if st.button("🏠 Back to Home"):
    st.switch_page("Dashboard.py")