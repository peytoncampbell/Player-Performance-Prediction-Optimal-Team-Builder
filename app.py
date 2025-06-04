import streamlit as st
import pandas as pd
from pathlib import Path

POSITION_PRIORITY = ['SS', 'CF', 'C', '2B', '3B', 'RF', 'LF', '1B']
LEARN_MAPPING = {
    'C': 'LearnC',
    '1B': 'Learn1B',
    '2B': 'Learn2B',
    '3B': 'Learn3B',
    'SS': 'LearnSS',
    'LF': 'LearnLF',
    'CF': 'LearnCF',
    'RF': 'LearnRF',
}

@st.cache_data
def load_weights(path: Path) -> dict:
    df = pd.read_csv(path, index_col=0, header=None)
    weights = df.squeeze().to_dict()
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def best_offense_rating(row, weights: dict) -> float:
    return sum(row.get(stat, 0) * weight for stat, weight in weights.items())


def build_lineup(df: pd.DataFrame) -> pd.DataFrame:
    selected_titles = set()
    lineup_rows = []
    for pos in POSITION_PRIORITY:
        col = f'Rating_{pos}'
        learn_col = LEARN_MAPPING.get(pos)
        if col not in df.columns or learn_col not in df.columns:
            continue
        df_pos = df[(~df['//Card Title'].isin(selected_titles)) &
                    (df[learn_col] > 0) &
                    (df[col].notna())]
        if df_pos.empty:
            continue
        top_player = df_pos.sort_values(by=col, ascending=False).iloc[0].copy()
        top_player['Assigned_Position'] = pos
        top_player['Assigned_Value'] = top_player[col]
        selected_titles.add(top_player['//Card Title'])
        lineup_rows.append(top_player)

    df_remaining = df[~df['//Card Title'].isin(selected_titles)].copy()
    df_remaining['Rating_All_Positions'] = pd.to_numeric(df_remaining['Rating_All_Positions'], errors='coerce')
    top_dh = df_remaining.sort_values(by='Rating_All_Positions', ascending=False).head(1)
    if not top_dh.empty:
        dh_player = top_dh.iloc[0].copy()
        dh_player['Assigned_Position'] = 'DH'
        dh_player['Assigned_Value'] = dh_player['Rating_All_Positions']
        selected_titles.add(dh_player['//Card Title'])
        lineup_rows.append(dh_player)

    df_remaining = df[~df['//Card Title'].isin(selected_titles)].copy()
    backup_c = df_remaining[(df_remaining['Rating_C'].notna()) & (df_remaining['LearnC'] > 0)]
    backup_c = backup_c.sort_values(by='Rating_C', ascending=False).head(1)
    if not backup_c.empty:
        sub_c_player = backup_c.iloc[0].copy()
        sub_c_player['Assigned_Position'] = 'SUB_C'
        sub_c_player['Assigned_Value'] = sub_c_player['Rating_C']
        selected_titles.add(sub_c_player['//Card Title'])
        lineup_rows.append(sub_c_player)

    df_remaining = df[~df['//Card Title'].isin(selected_titles)].copy()
    df_remaining['INF_COVERAGE'] = df_remaining[['Learn1B', 'Learn2B', 'Learn3B', 'LearnSS']].gt(0).sum(axis=1)
    df_remaining = df_remaining[df_remaining['INF_COVERAGE'] > 0]
    sub1 = df_remaining.sort_values(by=['INF_COVERAGE', 'Rating_All_Positions'], ascending=False).head(1)
    if not sub1.empty:
        sub1_player = sub1.iloc[0].copy()
        sub1_player['Assigned_Position'] = 'SUB1'
        sub1_player['Assigned_Value'] = sub1_player['Rating_All_Positions']
        selected_titles.add(sub1_player['//Card Title'])
        lineup_rows.append(sub1_player)

    df_remaining = df[~df['//Card Title'].isin(selected_titles)].copy()
    df_remaining['OF_COVERAGE'] = df_remaining[['LearnLF', 'LearnCF', 'LearnRF']].gt(0).sum(axis=1)
    df_remaining = df_remaining[df_remaining['OF_COVERAGE'] > 0]
    sub2 = df_remaining.sort_values(by=['OF_COVERAGE', 'Rating_All_Positions'], ascending=False).head(1)
    if not sub2.empty:
        sub2_player = sub2.iloc[0].copy()
        sub2_player['Assigned_Position'] = 'SUB2'
        sub2_player['Assigned_Value'] = sub2_player['Rating_All_Positions']
        selected_titles.add(sub2_player['//Card Title'])
        lineup_rows.append(sub2_player)

    df_remaining = df[~df['//Card Title'].isin(selected_titles)].copy()
    sub3 = df_remaining.sort_values(by='Rating_All_Positions', ascending=False).head(1)
    if not sub3.empty:
        sub3_player = sub3.iloc[0].copy()
        sub3_player['Assigned_Position'] = 'SUB3'
        sub3_player['Assigned_Value'] = sub3_player['Rating_All_Positions']
        selected_titles.add(sub3_player['//Card Title'])
        lineup_rows.append(sub3_player)

    df_lineup = pd.DataFrame(lineup_rows)
    sort_order = {pos: i for i, pos in enumerate(POSITION_PRIORITY + ['DH', 'SUB_C', 'SUB1', 'SUB2', 'SUB3'])}
    df_lineup['SortKey'] = df_lineup['Assigned_Position'].map(sort_order)
    df_lineup = df_lineup.sort_values(by='SortKey').drop(columns=['SortKey'])
    return df_lineup


def suggest_upgrades(lineup: pd.DataFrame, df_pool: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df_pool = df_pool.copy()
    df_pool['Rating_All_Positions'] = pd.to_numeric(df_pool['Rating_All_Positions'], errors='coerce')
    df_pool['Best_Offense_Rating'] = df_pool.apply(lambda r: best_offense_rating(r, weights), axis=1)
    upgrades = []
    selected_titles = set(lineup['//Card Title'])
    for _, row in lineup.iterrows():
        pos = row['Assigned_Position']
        current_card = row['//Card Title']
        current_value = row['Assigned_Value']
        current_price = row.get('Sell Order Low', None)
        if pos in ['DH', 'SUB1', 'SUB2', 'SUB3']:
            upgrade_col = 'Rating_All_Positions'
            df_upg = df_pool.copy()
            if pos == 'SUB1':
                df_upg = df_upg[(df_upg[['Learn1B', 'Learn2B', 'Learn3B', 'LearnSS']] > 0).all(axis=1)]
            elif pos == 'SUB2':
                df_upg = df_upg[(df_upg[['LearnLF', 'LearnCF', 'LearnRF']] > 0).all(axis=1)]
        elif pos == 'SUB_C':
            upgrade_col = 'Rating_C'
            df_upg = df_pool[(df_pool['Rating_C'].notna()) & (df_pool['LearnC'] > 0)]
        else:
            upgrade_col = f'Rating_{pos}'
            learn_col = LEARN_MAPPING.get(pos)
            df_upg = df_pool[df_pool[learn_col] > 0]
        df_better = df_upg[(df_upg[upgrade_col] > current_value) & (~df_upg['//Card Title'].isin(selected_titles))]
        if df_better.empty:
            continue
        df_better['Value_vs_Price'] = ((df_better[upgrade_col] - current_value) / df_better['Sell Order Low']) * 1000
        best = df_better.sort_values(by='Value_vs_Price', ascending=False).iloc[0]
        upgrades.append({
            'Position': pos,
            'Current Player': current_card,
            'Current Value': current_value,
            'Current Price': current_price,
            'Upgrade Player': best['//Card Title'],
            'Upgrade Value': best[upgrade_col],
            'Upgrade Price': best['Sell Order Low'],
            'Value_vs_Price': best['Value_vs_Price'],
        })
        selected_titles.add(best['//Card Title'])
    return pd.DataFrame(upgrades).sort_values(by='Value_vs_Price', ascending=False)


def main():
    st.title('Perfect Team Lineup Builder')
    uploaded_file = st.file_uploader('Upload your card list CSV', type='csv')
    if uploaded_file is None:
        st.info('Upload a CSV file generated by your card list export.')
        return
    df = pd.read_csv(uploaded_file)
    if 'owned' not in df.columns:
        st.error('CSV must contain an "owned" column to indicate which cards you have.')
        return
    weights = load_weights(Path('output/learned_weights_All_Positions.csv'))
    df_owned = df[df['owned'] == 1].copy()
    lineup = build_lineup(df_owned)
    st.subheader('Best Lineup')
    st.dataframe(lineup[['Assigned_Position', '//Card Title', 'Assigned_Value']])
    upgrades = suggest_upgrades(lineup, df[df['Sell Order Low'] > 0].copy(), weights)
    st.subheader('Suggested Upgrades')
    st.dataframe(upgrades)


if __name__ == '__main__':
    main()
