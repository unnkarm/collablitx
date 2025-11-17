import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from textblob import TextBlob
import pandas as pd
from collections import defaultdict
import statistics
import os

# -------------------
# Configuration
# -------------------
st.set_page_config(
    page_title="Advanced Peer Review Platform",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .review-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

REVIEWS_FILE = "reviews.json"

# -------------------
# Helper Functions
# -------------------
def load_reviews():
    if not os.path.exists(REVIEWS_FILE):
        return []
    try:
        with open(REVIEWS_FILE, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            # Normalize entries: ensure date, timestamp, id, rating types
            for i, r in enumerate(data):
                if "date" not in r or not r.get("date"):
                    # try to infer from timestamp or set today
                    if r.get("timestamp"):
                        try:
                            dt = datetime.fromisoformat(r["timestamp"])
                            r["date"] = dt.strftime("%Y-%m-%d")
                        except Exception:
                            r["date"] = datetime.now().strftime("%Y-%m-%d")
                    else:
                        r["date"] = datetime.now().strftime("%Y-%m-%d")
                if "timestamp" not in r or not r.get("timestamp"):
                    r["timestamp"] = datetime.now().isoformat()
                if "id" not in r:
                    r["id"] = i + 1
                # Ensure rating is numeric
                try:
                    r["rating"] = float(r.get("rating", 0))
                except Exception:
                    r["rating"] = 0.0
                # Ensure skills exist
                if "skills" not in r or not isinstance(r["skills"], dict):
                    r["skills"] = {}
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_reviews(reviews):
    with open(REVIEWS_FILE, "w") as f:
        json.dump(reviews, f, indent=4)

def analyze_sentiment(text):
    blob = TextBlob(text or "")
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.3:
        label = "Very Positive"
    elif polarity > 0.1:
        label = "Positive"
    elif polarity < -0.3:
        label = "Very Negative"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return label, polarity, subjectivity

def calculate_user_stats(reviews, username):
    user_reviews = [r for r in reviews if r.get("reviewee") == username]
    if not user_reviews:
        return None

    ratings = [float(r.get("rating", 0)) for r in user_reviews]
    skills = defaultdict(list)
    for r in user_reviews:
        for skill, value in (r.get("skills") or {}).items():
            try:
                skills[skill].append(float(value))
            except Exception:
                pass

    avg_skills = {skill: statistics.mean(values) for skill, values in skills.items()} if skills else {}
    sentiment_scores = [r.get("sentiment_score", 0) for r in user_reviews]

    return {
        "total_reviews": len(user_reviews),
        "avg_rating": statistics.mean(ratings) if ratings else 0,
        "rating_std": statistics.stdev(ratings) if len(ratings) > 1 else 0,
        "avg_skills": avg_skills,
        "avg_sentiment": statistics.mean(sentiment_scores) if sentiment_scores else 0,
        "recent_trend": calculate_trend(user_reviews),
        "strengths": identify_strengths(avg_skills),
        "areas_for_improvement": identify_improvements(avg_skills)
    }

def calculate_trend(user_reviews):
    if len(user_reviews) < 2:
        return "Insufficient data"
    # Sort by timestamp safely
    def safe_ts(r):
        try:
            return r.get("timestamp") or ""
        except Exception:
            return ""
    sorted_reviews = sorted(user_reviews, key=safe_ts)
    recent = sorted_reviews[-3:]
    older = sorted_reviews[:-3] if len(sorted_reviews) > 3 else sorted_reviews[:1]
    try:
        recent_avg = statistics.mean([float(r.get("rating", 0)) for r in recent])
        older_avg = statistics.mean([float(r.get("rating", 0)) for r in older])
    except Exception:
        return "Insufficient data"
    diff = recent_avg - older_avg
    if diff > 0.3:
        return "📈 Improving"
    elif diff < -0.3:
        return "📉 Declining"
    else:
        return "➡️ Stable"

def identify_strengths(skills):
    if not skills:
        return []
    sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, score in sorted_skills[:2] if score >= 4.0]

def identify_improvements(skills):
    if not skills:
        return []
    sorted_skills = sorted(skills.items(), key=lambda x: x[1])
    return [skill for skill, score in sorted_skills[:2] if score < 3.5]

def create_radar_chart(skills_data, title):
    if not skills_data:
        # empty chart placeholder
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    categories = list(skills_data.keys())
    values = list(skills_data.values())

    # close the loop for radar if necessary (Plotly handles without explicit closure)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Skills',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        title=title,
        height=400
    )
    return fig

def create_rating_distribution(reviews):
    if not reviews:
        fig = go.Figure()
        fig.update_layout(title="Rating Distribution (No data)")
        return fig
    ratings = [float(r.get("rating", 0)) for r in reviews]
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    fig = go.Figure(data=[go.Bar(x=rating_counts.index, y=rating_counts.values)])
    fig.update_layout(title="Rating Distribution", xaxis_title="Rating", yaxis_title="Count", height=300)
    return fig

def create_timeline_chart(reviews, user=None):
    if user:
        reviews = [r for r in reviews if r.get("reviewee") == user]
    if not reviews:
        fig = go.Figure()
        fig.update_layout(title=f"Rating Trend Over Time{' - ' + user if user else ''} (No data)")
        return fig

    df = pd.DataFrame(reviews)
    # tolerant parsing of dates
    df['date'] = pd.to_datetime(df.get('date', None), errors='coerce')
    df = df.dropna(subset=['date'])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Rating Trend Over Time{' - ' + user if user else ''} (No valid dates)")
        return fig
    df = df.sort_values('date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rating'].astype(float),
        mode='lines+markers',
        name='Rating',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"Rating Trend Over Time{' - ' + user if user else ''}",
        xaxis_title="Date",
        yaxis_title="Rating",
        yaxis=dict(range=[0, 5.5]),
        height=400
    )
    return fig

# -------------------
# Initialize Session State
# -------------------
if 'reviews' not in st.session_state:
    st.session_state.reviews = load_reviews()

if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False

# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.title("🎯 Navigation")
    all_reviewees = sorted(list({r.get("reviewee", "N/A") for r in st.session_state.reviews}))
    st.markdown("---")
    st.subheader("Quick Stats")

    if st.session_state.reviews:
        total_reviews = len(st.session_state.reviews)
        try:
            avg_rating = statistics.mean([float(r.get("rating", 0)) for r in st.session_state.reviews])
        except Exception:
            avg_rating = 0
        st.metric("Total Reviews", total_reviews)
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        st.metric("Active Users", len(all_reviewees))

        # Recent activity
        st.markdown("---")
        st.subheader("Recent Activity")
        recent = sorted(st.session_state.reviews, key=lambda x: x.get("timestamp", ""), reverse=True)[:3]
        for r in recent:
            st.caption(f"⭐ {r.get('rating', 'N/A')} - {r.get('reviewee', 'N/A')} ({r.get('date','N/A')})")
    else:
        st.info("No reviews yet")

    st.markdown("---")
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("Confirm deletion"):
            st.session_state.reviews = []
            save_reviews([])
            st.rerun()

# -------------------
# Main Content
# -------------------
st.title("🌟 Advanced Peer Review Platform")
st.markdown("*Comprehensive feedback system with analytics and insights*")

tabs = st.tabs(["📝 Submit Review", "📊 View Reviews", "📈 Analytics Dashboard", "👤 User Profiles"])

# -------------------
# Tab 1: Submit Review
# -------------------
with tabs[0]:
    st.header("Submit a New Review")
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("review_form", clear_on_submit=True):
            st.subheader("Review Details")
            form_col1, form_col2 = st.columns(2)

            with form_col1:
                reviewer = st.text_input("Your Name", placeholder="Enter your name (optional if Anonymous)")
                reviewee_options = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
                reviewee = st.selectbox("Reviewee *", reviewee_options)
                project = st.text_input("Project Name", placeholder="Optional")
                review_type = st.selectbox("Review Type", ["Peer Review", "Project Review", "Performance Review", "Code Review"])

            with form_col2:
                category = st.selectbox("Category", ["Development", "Design", "Management", "Research", "Other"])
                confidential = st.checkbox("Mark as confidential")
                anonymous = st.checkbox("Submit anonymously")
                priority = st.select_slider("Priority", ["Low", "Medium", "High"])

            st.markdown("---")
            st.subheader("Skills Assessment")

            skill_col1, skill_col2, skill_col3 = st.columns(3)

            with skill_col1:
                rating = st.slider("Overall Rating ⭐", 1, 5, 3)
                comm = st.slider("Communication 💬", 1, 5, 3)
                tech = st.slider("Technical Skills 💻", 1, 5, 3)

            with skill_col2:
                team = st.slider("Teamwork 🤝", 1, 5, 3)
                problem = st.slider("Problem Solving 🧩", 1, 5, 3)
                creativity = st.slider("Creativity 💡", 1, 5, 3)

            with skill_col3:
                leadership = st.slider("Leadership 👑", 1, 5, 3)
                initiative = st.slider("Initiative 🚀", 1, 5, 3)
                reliability = st.slider("Reliability ⏰", 1, 5, 3)

            st.markdown("---")
            st.subheader("Detailed Feedback")

            comment = st.text_area(
                "Comment *",
                placeholder="Provide detailed feedback about strengths, areas for improvement, and specific examples...",
                height=150
            )

            strengths = st.text_area("Key Strengths", placeholder="List notable strengths and achievements")
            improvements = st.text_area("Areas for Development", placeholder="Suggest specific areas for growth")

            # Simplified validation: only require reviewee and comment
            # Reviewer can be empty if anonymous is checked
            submitted = st.form_submit_button("🚀 Submit Review", use_container_width=True)

            if submitted:
                # Validation check on submission
                if not comment.strip():
                    st.error("⚠️ Please provide a comment before submitting.")
                elif not anonymous and not reviewer.strip():
                    st.error("⚠️ Please enter your name or check 'Submit anonymously'.")
                else:
                    sentiment_label, polarity, subjectivity = analyze_sentiment(comment)
                    reviewer_name = "Anonymous" if anonymous or not reviewer.strip() else reviewer.strip()

                    new_review = {
                        "id": (len(st.session_state.reviews) + 1),
                        "reviewer": reviewer_name,
                        "reviewee": reviewee.strip(),
                        "project": project.strip(),
                        "rating": float(rating),
                        "skills": {
                            "communication": int(comm),
                            "technical": int(tech),
                            "teamwork": int(team),
                            "problemSolving": int(problem),
                            "creativity": int(creativity),
                            "leadership": int(leadership),
                            "initiative": int(initiative),
                            "reliability": int(reliability)
                        },
                        "comment": comment.strip(),
                        "strengths": strengths.strip(),
                        "improvements": improvements.strip(),
                        "sentiment": sentiment_label,
                        "sentiment_score": float(polarity),
                        "subjectivity": float(subjectivity),
                        "review_type": review_type,
                        "category": category,
                        "confidential": bool(confidential),
                        "priority": priority,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "timestamp": datetime.now().isoformat()
                    }

                    st.session_state.reviews.append(new_review)
                    save_reviews(st.session_state.reviews)

                    st.success(f"✅ Review submitted successfully for {reviewee}!")
                    st.balloons()

    with col2:
        st.info("💡 **Tips for Great Reviews**\n\n"
                "• Be specific with examples\n"
                "• Balance positive and constructive feedback\n"
                "• Focus on behaviors, not personality\n"
                "• Provide actionable suggestions\n"
                "• Be respectful and professional")

# -------------------
# Tab 2: View Reviews
# -------------------
with tabs[1]:
    st.header("Review Repository")
    reviews = st.session_state.reviews

    if not reviews:
        st.info("📭 No reviews submitted yet. Be the first to submit one!")
    else:
        # Filters
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            filter_user = st.selectbox("👤 Reviewee", ["All"] + sorted(list({r.get("reviewee", "N/A") for r in reviews})))

        with filter_col2:
            filter_type = st.selectbox("📋 Type", ["All"] + sorted(list({r.get("review_type", "Peer Review") for r in reviews})))

        with filter_col3:
            filter_sentiment = st.selectbox("😊 Sentiment", ["All", "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"])

        with filter_col4:
            sort_option = st.selectbox("🔄 Sort By", [
                "Date (Newest)", "Date (Oldest)",
                "Rating (High → Low)", "Rating (Low → High)",
                "Sentiment (Positive → Negative)"
            ])

        # Apply filters
        filtered_reviews = reviews.copy()

        if filter_user != "All":
            filtered_reviews = [r for r in filtered_reviews if r.get("reviewee") == filter_user]

        if filter_type != "All":
            filtered_reviews = [r for r in filtered_reviews if r.get("review_type", "Peer Review") == filter_type]

        if filter_sentiment != "All":
            filtered_reviews = [r for r in filtered_reviews if r.get("sentiment") == filter_sentiment]

        # Apply sorting safely
        if sort_option == "Date (Newest)":
            filtered_reviews.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        elif sort_option == "Date (Oldest)":
            filtered_reviews.sort(key=lambda x: x.get("timestamp", ""))
        elif sort_option == "Rating (High → Low)":
            filtered_reviews.sort(key=lambda x: float(x.get("rating", 0)), reverse=True)
        elif sort_option == "Rating (Low → High)":
            filtered_reviews.sort(key=lambda x: float(x.get("rating", 0)))
        elif sort_option == "Sentiment (Positive → Negative)":
            filtered_reviews.sort(key=lambda x: float(x.get("sentiment_score", 0)), reverse=True)

        st.markdown(f"**Showing {len(filtered_reviews)} of {len(reviews)} reviews**")
        st.markdown("---")

        # Display reviews
        for r in filtered_reviews:
            sentiment_emoji = {
                "Very Positive": "😄", "Positive": "🙂",
                "Neutral": "😐", "Negative": "😟", "Very Negative": "😢"
            }.get(r.get("sentiment", "Neutral"), "😐")

            with st.expander(
                f"⭐ {r.get('rating','N/A')}/5 | {r.get('reviewee','N/A')} ← {r.get('reviewer','N/A')} | "
                f"{r.get('review_type', 'Peer Review')} | {r.get('date','N/A')} {sentiment_emoji}"
            ):
                review_col1, review_col2 = st.columns([2, 1])

                with review_col1:
                    if r.get("project"):
                        st.markdown(f"**📁 Project:** {r.get('project')}")
                    st.markdown("**💬 Feedback:**")
                    st.write(r.get('comment', ''))

                    if r.get("strengths"):
                        st.markdown("**✨ Strengths:**")
                        st.write(r.get('strengths'))

                    if r.get("improvements"):
                        st.markdown("**🎯 Development Areas:**")
                        st.write(r.get('improvements'))

                    st.markdown(
                        f"**Sentiment:** {sentiment_emoji} {r.get('sentiment', 'N/A')} "
                        f"(Score: {float(r.get('sentiment_score', 0)):.2f}, "
                        f"Subjectivity: {float(r.get('subjectivity', 0)):.2f})"
                    )

                with review_col2:
                    st.markdown("**Skills Breakdown:**")
                    skills = r.get("skills", {}) or {}
                    for skill, value in skills.items():
                        try:
                            val = float(value)
                        except Exception:
                            val = 0.0
                        st.progress(min(max(val / 5.0, 0.0), 1.0))
                        st.caption(f"{skill.capitalize()}: {val}/5")

                    if r.get("confidential"):
                        st.warning("🔒 Confidential")

                    if r.get("priority"):
                        priority_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        st.info(f"{priority_colors.get(r.get('priority'), '⚪')} Priority: {r.get('priority')}")

# -------------------
# Tab 3: Analytics Dashboard
# -------------------
with tabs[2]:
    st.header("Analytics Dashboard")

    if not st.session_state.reviews:
        st.info("📊 No data available yet. Submit reviews to see analytics.")
    else:
        # Overall metrics
        st.subheader("📈 Overall Performance Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        all_ratings = [float(r.get("rating", 0)) for r in st.session_state.reviews]
        all_sentiments = [float(r.get("sentiment_score", 0)) for r in st.session_state.reviews]

        with metric_col1:
            st.metric("Total Reviews", len(st.session_state.reviews))

        with metric_col2:
            st.metric("Average Rating", f"{statistics.mean(all_ratings):.2f}" if all_ratings else "N/A")

        with metric_col3:
            st.metric("Rating Std Dev", f"{statistics.stdev(all_ratings) if len(all_ratings) > 1 else 0:.2f}")

        with metric_col4:
            st.metric("Avg Sentiment", f"{statistics.mean(all_sentiments):.2f}" if all_sentiments else "0.00")

        st.markdown("---")

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(create_rating_distribution(st.session_state.reviews), use_container_width=True)

        with chart_col2:
            sentiments = [r.get("sentiment", "Neutral") for r in st.session_state.reviews]
            sentiment_df = pd.DataFrame({"Sentiment": sentiments})
            sentiment_counts = sentiment_df["Sentiment"].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Timeline
        st.plotly_chart(create_timeline_chart(st.session_state.reviews), use_container_width=True)

        st.markdown("---")

        # Skills comparison across all users
        st.subheader("🎯 Skills Comparison Across Team")
        all_reviewees = sorted(list({r.get("reviewee", "N/A") for r in st.session_state.reviews}))

        if all_reviewees:
            skills_comparison = {}
            for user in all_reviewees:
                user_reviews = [r for r in st.session_state.reviews if r.get("reviewee") == user]
                skills = defaultdict(list)
                for r in user_reviews:
                    for skill, value in (r.get("skills") or {}).items():
                        try:
                            skills[skill].append(float(value))
                        except Exception:
                            pass
                skills_comparison[user] = {skill: statistics.mean(values) for skill, values in skills.items()} if skills else {}

            # safe skill names
            first_vals = next((v for v in skills_comparison.values() if v), {})
            skill_names = list(first_vals.keys()) if first_vals else []

            fig = go.Figure()
            for user in all_reviewees:
                fig.add_trace(go.Bar(
                    name=user,
                    x=skill_names,
                    y=[skills_comparison[user].get(skill, 0) for skill in skill_names]
                ))

            fig.update_layout(
                barmode='group',
                title="Team Skills Comparison",
                xaxis_title="Skills",
                yaxis_title="Average Rating",
                yaxis=dict(range=[0, 5]),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No users to compare yet.")

# -------------------
# Tab 4: User Profiles
# -------------------
with tabs[3]:
    st.header("User Profiles")
    if not st.session_state.reviews:
        st.info("👤 No user data available yet.")
    else:
        all_reviewees = sorted(list({r.get("reviewee", "N/A") for r in st.session_state.reviews}))
        selected_user = st.selectbox("Select User", all_reviewees)

        if selected_user:
            stats = calculate_user_stats(st.session_state.reviews, selected_user)
            if stats:
                st.markdown(f"## 👤 {selected_user}'s Profile")
                profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

                with profile_col1:
                    st.metric("Total Reviews", stats["total_reviews"])
                with profile_col2:
                    st.metric("Average Rating", f"{stats['avg_rating']:.2f} ⭐")
                with profile_col3:
                    st.metric("Consistency", f"±{stats['rating_std']:.2f}")
                with profile_col4:
                    st.metric("Trend", stats["recent_trend"])

                st.markdown("---")
                radar_col, timeline_col = st.columns(2)

                with radar_col:
                    st.plotly_chart(
                        create_radar_chart(stats.get("avg_skills", {}), f"{selected_user}'s Skills Profile"),
                        use_container_width=True
                    )

                with timeline_col:
                    st.plotly_chart(create_timeline_chart(st.session_state.reviews, selected_user), use_container_width=True)

                st.markdown("---")
                strength_col, improve_col = st.columns(2)

                with strength_col:
                    st.subheader("💪 Key Strengths")
                    if stats.get("strengths"):
                        for strength in stats["strengths"]:
                            st.success(f"✓ {strength.capitalize()}: {stats['avg_skills'][strength]:.2f}/5")
                    else:
                        st.info("Keep up the great work!")

                with improve_col:
                    st.subheader("🎯 Development Opportunities")
                    if stats.get("areas_for_improvement"):
                        for area in stats["areas_for_improvement"]:
                            st.warning(f"→ {area.capitalize()}: {stats['avg_skills'][area]:.2f}/5")
                    else:
                        st.info("Excellent performance across all areas!")

                st.markdown("---")
                st.subheader("📝 Recent Reviews")
                user_reviews = [r for r in st.session_state.reviews if r.get("reviewee") == selected_user]
                user_reviews.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

                for r in user_reviews[:5]:
                    with st.expander(f"⭐ {r.get('rating','N/A')}/5 by {r.get('reviewer','N/A')} on {r.get('date','N/A')}"):
                        st.write(r.get('comment', ''))
                        if r.get('strengths'):
                            st.markdown(f"**Strengths:** {r.get('strengths')}")
                        if r.get('improvements'):
                            st.markdown(f"**Improvements:** {r.get('improvements')}")