# src/ranking_logic.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def rank_proposals(csv_file: str = "data/proposals.csv"):
    df = pd.read_csv(csv_file)

    scaler = MinMaxScaler()
    df[['relevance_score', 'rating', 'success_rate']] = scaler.fit_transform(
        df[['relevance_score', 'rating', 'success_rate']]
    )

    if df['bid_price'].nunique() > 1:
        df['bid_price'] = (df['bid_price'].max() - df['bid_price']) / (
            df['bid_price'].max() - df['bid_price'].min()
        )
    else:
        df['bid_price'] = 1.0

    weights = {
        'relevance_score': 0.4, 'rating': 0.25,
        'success_rate': 0.25, 'bid_price': 0.1
    }

    df['final_score'] = sum(df[col] * weight for col, weight in weights.items())

    return df.sort_values(by='final_score', ascending=False)