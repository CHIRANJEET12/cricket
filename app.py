import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

st.set_page_config(page_title="Cricket Prediction", page_icon="🏏", layout="wide")

@st.cache_data
def load_playe_elo_data():
    """Load trained match prediction model"""
    try:
        with open('model.pkl','rb') as f:
            playe_elo_pred = pickle.load(f)
        return playe_elo_pred
    except FileNotFoundError:
        st.error("Player Elo Rating model file not found!")
        return None
    
@st.cache_data
def load_match_win_data():
    """Load trained match_win prediction model"""
    try:
        with open('pipe1.pkl','rb') as f:
            matchwin = pickle.load(f)
        delivery_data = pd.read_csv("delivery_df1.csv")
        return matchwin,delivery_data
    except FileNotFoundError:
        st.error("Player Match win file not found!")
        return None
    
@st.cache_data
def load_match_model():
    """Load Player stats for Comparison"""
    try:
        with open('processed_cricket_data.pkl','rb') as f:
            match_pred = pickle.load(f)
        return match_pred
    except FileNotFoundError:
        st.error("Match pred file not found!")
        return None

@st.cache_data
def load_player_data():
    """Load player statistics for comparisons"""
    try:
        player_data = pd.read_csv("players_pred.csv")
        return player_data
    except FileNotFoundError:
        st.error("Player comp file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return None

def find_player_name_column(df):
    """Find the correct column name for player names"""
    possible_names = ['player_name', 'Player_Name', 'player', 'Player', 'name', 'Name', 'batsman', 'Batsman']
    for col in possible_names:
        if col in df.columns:
            return col
    # If no standard name found, return the first column
    return df.columns[0]

def main():
    st.header("🏏 Cricket Prediction Analytics")

    player_data = load_player_data()
    player_elo_pred = load_playe_elo_data()
    match_pred,delivery_data = load_match_win_data()

    # Debug: Check what we loaded
    if player_data is not None:
        st.sidebar.write("📊 Data Loaded Successfully!")
        st.sidebar.write(f"Shape: {player_data.shape}")
        st.sidebar.write("Columns:", player_data.columns.tolist())
        
        # Find the correct player name column
        player_name_col = find_player_name_column(player_data)
        st.sidebar.write(f"Using column '{player_name_col}' for player names")
        
        # Show sample player names
        if player_name_col in player_data.columns:
            st.sidebar.write("Sample players:", player_data[player_name_col].head(5).tolist())

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Feature:",
        ["🏠 Home", "👥 Player Comparison", "📊 Match Prediction"]
    )

    if page == "👥 Player Comparison":
        player_comparison_page(player_data, player_elo_pred)
    elif page == "📊 Match Prediction":
        match_prediction_page(match_pred,delivery_data)
    else:  
        home_page(player_data, match_pred)

def home_page(player_data, match_pred):
    st.subheader("Welcome to Cricket Prediction Analytics!")

    if player_data is not None:
        # Find the correct player name column
        player_name_col = find_player_name_column(player_data)
        if player_name_col in player_data.columns:
            player_count = len(player_data[player_name_col].unique())
        else:
            player_count = "N/A"
    else:
        player_count = "N/A"

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Players Available", player_count)
    with col2:
        st.metric("Model Status", "Loaded" if match_pred is not None else "Not Available")

    st.write("""
    ### Features Available:
    - **Player Comparison**: Compare batting, bowling, and overall statistics between two players
    - **Match Prediction**: Predict match outcomes based on current match situation
    - **Data Analytics**: Explore cricket statistics and trends
    """)

def enhanced_compare_players(player1, player2, df):
    # Find the correct player name column
    player_name_col = find_player_name_column(df)
    
    try:
        p1 = df[df[player_name_col] == player1].iloc[0]
        p2 = df[df[player_name_col] == player2].iloc[0]
    except IndexError:
        st.error(f"One or both players not found in dataset")
        return

    metrics = {
        'Batting': ['total_runs', 'average', 'strikerate'],
        'Bowling': ['wickets', 'bowling_economy', 'bowling_strike_rate'],
        'Overall': ['batting_elo', 'bowling_elo', 'overall_elo']
    }
    
    # Initialize counters
    total_metrics = 9  # 3 categories × 3 metrics each
    player1_wins = 0
    player2_wins = 0
    ties = 0
    
    comparison_data = []
    
    # Create radar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (category, feature_list) in enumerate(metrics.items()):
        values1 = [p1[f] for f in feature_list]
        values2 = [p2[f] for f in feature_list]
        
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform([values1, values2])
        
        angles = np.linspace(0, 2*np.pi, len(feature_list), endpoint=False).tolist()
        angles += angles[:1]  
        scaled_values1 = list(scaled_values[0]) + [scaled_values[0][0]]
        scaled_values2 = list(scaled_values[1]) + [scaled_values[1][0]]
        
        ax = axes[idx]
        ax.plot(angles, scaled_values1, 'o-', linewidth=2, label=player1)
        ax.plot(angles, scaled_values2, 'o-', linewidth=2, label=player2)
        ax.fill(angles, scaled_values1, alpha=0.25)
        ax.fill(angles, scaled_values2, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_list)
        ax.set_title(f'{category} Comparison')
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display comparison table and count wins
    st.subheader("📊 Player Comparison Results")
    
    for category, feature_list in metrics.items():
        st.write(f"**{category}:**")
        category_data = []
        
        for feature in feature_list:
            val1, val2 = p1[feature], p2[feature]
            
            # Determine winner for this metric
            if val1 > val2:
                winner = player1
                player1_wins += 1
                winner_icon = "✅"
            elif val2 > val1:
                winner = player2
                player2_wins += 1
                winner_icon = "✅"
            else:
                winner = "Tie"
                ties += 1
                winner_icon = "⚖️"
            
            category_data.append({
                'Metric': feature,
                player1: f"{val1:.2f}",
                player2: f"{val2:.2f}",
                'Winner': f"{winner_icon} {winner}"
            })
        
        # Display category comparison
        category_df = pd.DataFrame(category_data)
        st.dataframe(category_df, use_container_width=True)
        st.write("")
        
        # Add to overall comparison data
        comparison_data.extend(category_data)
    
    # Display overall statistics
    st.subheader("🏆 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(f"{player1} Wins", player1_wins)
    with col2:
        st.metric(f"{player2} Wins", player2_wins)
    with col3:
        st.metric("Ties", ties)
    with col4:
        st.metric("Total Metrics", total_metrics)
    
    # Determine and display overall winner
    st.subheader("🎯 Final Verdict")
    
    if player1_wins > player2_wins:
        win_percentage = (player1_wins / total_metrics) * 100
        st.success(f"🏆 **{player1} is the OVERALL WINNER!**")
        st.info(f"Won {player1_wins} out of {total_metrics} metrics ({win_percentage:.1f}%)")
        st.balloons()
    elif player2_wins > player1_wins:
        win_percentage = (player2_wins / total_metrics) * 100
        st.success(f"🏆 **{player2} is the OVERALL WINNER!**")
        st.info(f"Won {player2_wins} out of {total_metrics} metrics ({win_percentage:.1f}%)")
        st.balloons()
    else:
        st.info("🤝 **It's a TIE!** Both players are equally matched across all metrics.")
    
    # Progress bars for visual comparison
    st.subheader("📈 Win Distribution")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.write(f"**{player1}**")
        st.progress(player1_wins / total_metrics)
        st.write(f"{player1_wins}/{total_metrics} metrics")
    
    with col6:
        st.write(f"**{player2}**")
        st.progress(player2_wins / total_metrics)
        st.write(f"{player2_wins}/{total_metrics} metrics")
def player_comparison_page(player_data, player_elo_pred):
    st.subheader("👥 Player Comparison")   

    if player_data is None:
        st.error("Player data not loaded. Please check if players_pred.csv exists.")
        return 

    # Find the correct player name column
    player_name_col = find_player_name_column(player_data)
    
    if player_name_col not in player_data.columns:
        st.error(f"Could not find player name column in the dataset. Available columns: {player_data.columns.tolist()}")
        return

    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox("Select player 1", player_data[player_name_col].unique(), key="player1")

    with col2:
        avail_players = [p for p in player_data[player_name_col].unique() if p != player1]
        player2 = st.selectbox("Select player 2", avail_players, key="player2")
            
    if st.button("Compare Players", type="primary"):
        if player1 and player2:
            with st.spinner("Comparing players..."):
                enhanced_compare_players(player1, player2, player_data)
        else:
            st.warning("Please select both players!")


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
def match_prediction_page(match_pred, delivery_data):
    st.subheader("📊 Match Prediction")
    
    if match_pred is None:
        st.error("Match prediction model not loaded!")
        return
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["🎯 Live Prediction", "📈 Match Progression"])
    
    with tab1:
        st.write("Enter match details to predict the outcome:")

        col1, col2 = st.columns(2)

        with col1:
            batting_team = st.selectbox('Select the batting team', sorted(teams), key="batting_pred")
        with col2:
            avail_bowling_teams = [team for team in sorted(teams) if team != batting_team]
            bowling_team = st.selectbox('Select the bowling team', avail_bowling_teams, key="bowling_pred")

        selected_city = st.selectbox('Select host city', sorted(cities), key="city_pred")

        target = st.number_input('Target', min_value=1, max_value=400, value=150, key="target_pred")

        col3, col4, col5 = st.columns(3)
        
        with col3:
            score = st.number_input('Score', min_value=0, max_value=400, value=0, key="score_pred")
        with col4:
            overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="overs_pred")
        with col5:
            wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, value=0, key="wickets_pred")

        if st.button("Predict Probability", key="predict_btn"):
            if overs >= 20.0 or wickets_out >= 10:
                st.error("Innings completed! Cannot predict further.")
                return
                
            if score >= target:
                st.success(f"{batting_team} has already won the match!")
                return

            runs_left = target - score
            balls_left = max(0, 120 - (overs * 6))
            wickets_left = 10 - wickets_out
            
            crr = score / overs if overs > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets_left],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            try:
                result = match_pred.predict_proba(input_df)
                win_prob = result[0][1] * 100
                loss_prob = result[0][0] * 100
                
                st.success("Prediction Results:")
                col6, col7 = st.columns(2)
                with col6:
                    st.metric(f"{batting_team} Win Probability", f"{win_prob:.1f}%")
                with col7:
                    st.metric(f"{bowling_team} Win Probability", f"{loss_prob:.1f}%")
                    
                st.write("Win Probabilities:")
                st.progress(win_prob/100)
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with tab2:
        st.write("### Match Progression Analysis")
        
        if 'match_id' in delivery_data.columns:
            match_id = st.selectbox("Select any match id", delivery_data['match_id'].unique(), key="match_id")
            
            if st.button("Show Match Progression", key="progression_btn"):
                with st.spinner("Analyzing match progression..."):
                    temp_df, target = match_progression(delivery_data, match_id, match_pred)
                    
                    if not temp_df.empty:
                        st.success(f"Match Progression Analysis (Target: {target})")
                        
                        st.dataframe(temp_df, use_container_width=True)
                        
                        st.write("### Match Progression Chart")
                        fig, ax = plt.subplots(figsize=(18, 8))
                        
                        ax.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='yellow', linewidth=3, label='Wickets in Over')
                        ax.plot(temp_df['end_of_over'], temp_df['win'], color='#00a65a', linewidth=4, label='Win Probability %')
                        ax.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=4, label='Lose Probability %')
                        
                        ax.bar(temp_df['end_of_over'], temp_df['runs_after_over'], alpha=0.7, label='Runs in Over', color='skyblue')
                        
                        ax.set_xlabel('Over')
                        ax.set_ylabel('Values')
                        ax.set_title(f'Match Progression - Target: {target}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    else:
                        st.error("No progression data available for this match.")
        else:
            st.error("No match_id column found in delivery data")



if __name__ == "__main__":
    main()