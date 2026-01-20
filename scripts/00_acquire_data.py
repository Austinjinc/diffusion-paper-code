import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import scipy as sp
import matplotlib.ticker as mticker
import wrds
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

def smooth_iv_vega_paper(
    x, y, df, h1=0.002, h2=0.046,
    x_col='moneyness', y_col='ttm', iv_col='impl_volatility', vega_col='vega'
):
    dx = x - df[x_col].values
    dy = y - df[y_col].values
    weights = (1 / (2 * np.pi)) * np.exp(-0.5 * (dx ** 2 / h1) -0.5 * (dy ** 2 / h2))
    vega = df[vega_col].values
    iv = df[iv_col].values
    numerator = np.sum(weights * vega * iv)
    denominator = np.sum(weights * vega)
    return numerator / denominator

def get_year_data(year, conn):
    # Option price data
    df_year = conn.raw_sql(f"""
        SELECT
            op.date,
            op.exdate - op.date AS days,
            (op.exdate - op.date) / 365.0 AS ttm,
            op.vega,
            op.strike_price / 1000 AS strike_price,
            op.impl_volatility,
            op.cp_flag,
            s.close,
            op.volume,
            op.strike_price / (s.close * 1000) AS moneyness
        FROM optionm_all.opprcd{year} op
        JOIN optionm_all.secprd{year} s
            ON op.date = s.date
        WHERE
            op.secid = '108105'
            AND s.secid = '108105'
            AND op.volume > 0
            AND op.impl_volatility IS NOT NULL
            AND op.exdate - op.date >= 10
            AND op.exdate - op.date <= 400
            AND op.strike_price / (s.close * 1000) >= 0.5
            AND op.strike_price / (s.close * 1000) <= 1.5
            AND (
                (op.cp_flag = 'C' AND op.strike_price / 1000 > s.close)
                OR (op.cp_flag = 'P' AND op.strike_price / 1000 <= s.close)
                )
    """)
    df_year['date'] = pd.to_datetime(df_year['date'])
    df_year_sorted = df_year.sort_values('volume', ascending=False)
    df_unique = df_year_sorted.drop_duplicates(
        subset=['date', 'days', 'strike_price', 'cp_flag'],
        keep='first'
    )
    return df_unique

def acquire_and_clean_data():

    conn = wrds.Connection()
    
    all_dfs = []
    for year in range(1996, 2024): 
        print(f"Fetching OPPR data for year: {year}")
        try:
            df_unique = get_year_data(year, conn)
            all_dfs.append(df_unique)
        except Exception as e:
            print(f"Error in year {year}: {e}")

    df_all_years = pd.concat(all_dfs, ignore_index=True)

    m_grid = np.linspace(0.6, 1.4, 9)
    ttm_grid = np.array([1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1])
    grid_points = [(m, ttm) for ttm in ttm_grid for m in m_grid]

    all_results = []

    for date_val, group in df_all_years.groupby('date', sort=True):
        date_str = str(date_val)[:10]
        iv_row = [
            smooth_iv_vega_paper(
                x=m, y=ttm, df=group,
                x_col='moneyness', y_col='ttm',
                iv_col='impl_volatility', vega_col='vega'
            )
            for (m, ttm) in grid_points
        ]
        row = [date_str] + iv_row
        all_results.append(row)
    iv_surface_array = np.array(all_results)

    dates_array = iv_surface_array[:, 0]
    iv_data_flat = iv_surface_array[:, 1:]

    iv_data_reshaped = iv_data_flat.reshape(-1, 9, 9)
    iv_data_final = iv_data_reshaped.transpose(0, 2, 1)

    # 4. Save the two separate .npy files
    try:
        # np.save('iv_surfaces.npy', iv_data_final)
        # np.save('dates.npy', dates_array)
        
        print("\nSuccessfully saved 'iv_surfaces.npy' and 'dates.npy'")
        print(f"  - iv_surfaces.npy shape: {iv_data_final.shape}")
        print(f"  - dates.npy shape: {dates_array.shape}")

    except Exception as e:
        print(f"\nAn error occurred while saving: {e}")
    
    secid = '108105'
    years = range(1996, 2024)
    dfs = []

    for year in years:
        query = f"""
            SELECT date, close
            FROM optionm_all.secprd{year}
            WHERE secid = '{secid}'
        """
        df = conn.raw_sql(query)
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)

    prices['date'] = pd.to_datetime(prices['date'])
    prices.set_index('date', inplace=True)

    prices['returns'] = prices['close'].pct_change()
    prices['sq_returns'] = prices['returns']**2
    print("prices shape:", prices.shape)

    alpha_trend_short = 0.156  # For short-term trend (from lambda = 42.78)
    alpha_trend_long = 0.118   # For long-term trend (from lambda = 31.51)
    alpha_vol_short = 0.3   # For short-term volatility (from lambda = 3.694)
    alpha_vol_long = 0.15     # For long-term volatility (from lambda = 3.693)

    prices['EWMA_Trend_Short'] = prices['returns'].ewm(alpha=alpha_trend_short, adjust=False).mean()
    prices['EWMA_Trend_Long'] = prices['returns'].ewm(alpha=alpha_trend_long, adjust=False).mean()
    prices['EWMA_Vol_Short'] = prices['sq_returns'].ewm(alpha=alpha_vol_short, adjust=False).mean() * 100
    prices['EWMA_Vol_Long'] = prices['sq_returns'].ewm(alpha=alpha_vol_long, adjust=False).mean() * 100

    vix_df = pd.read_csv("data/VIX.csv")
    vix_df.dropna(subset=['observation_date', 'VIXCLS'], inplace=True)
    vix_df.rename(columns={'observation_date': 'date', 'VIXCLS': 'VIX_Close'}, inplace=True)
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_df.set_index('date', inplace=True)
    print(vix_df.dropna().shape)
    vix_df['VIX_Return'] = vix_df['VIX_Close'].pct_change(fill_method=None)

    prices = pd.concat([prices, vix_df['VIX_Return']], axis=1)
    final_features = prices.dropna()
    
    try:
        dates_from_surfaces = pd.to_datetime(dates_array)
        dates_from_features = final_features.index.normalize()
        common_dates = dates_from_surfaces.intersection(dates_from_features)

        print(f"Total dates from surfaces (dates_array): {len(dates_from_surfaces)}")
        print(f"Total dates from features (final_features): {len(dates_from_features)}")
        print(f"\nNumber of common dates found: {len(common_dates)}")

    except Exception as e:
        print(f"An error occurred while finding common dates: {e}")
        print("Please check that 'dates_array' and 'final_features' are defined correctly.")
        
    feature_columns = [
    'EWMA_Trend_Short', 
    'EWMA_Trend_Long', 
    'EWMA_Vol_Short', 
    'EWMA_Vol_Long',
    'VIX_Return'
    ]

    all_features = prices.dropna()
    final_features = all_features[feature_columns]

    print(f"Selected {len(feature_columns)} feature columns.")
    print(f"DataFrame shape after dropna and selection: {final_features.shape}")

    try:
        surface_dates_dt = pd.to_datetime(dates_array).normalize()
        surfaces_df = pd.DataFrame(
            data=iv_data_final.reshape(len(iv_data_final), -1), 
            index=surface_dates_dt
        )

        surfaces_df = surfaces_df[~surfaces_df.index.duplicated(keep='first')]
        features_df = final_features[~final_features.index.duplicated(keep='first')]

        aligned_surfaces_df, aligned_features_df = surfaces_df.align(
            features_df, 
            join='inner', 
            axis=0
        )

        print(f"Original surfaces: {len(surfaces_df)} days")
        print(f"Original features: {len(features_df)} days")
        print(f"Aligned (common): {len(aligned_surfaces_df)} days")

        final_dates = aligned_surfaces_df.index.strftime('%Y-%m-%d').to_numpy()
        final_surfaces = aligned_surfaces_df.to_numpy().reshape(-1, 9, 9)
        final_scalars = aligned_features_df.to_numpy()
        
        save_dir = os.path.join(
            os.path.dirname(__file__),  # This is the 'scripts' folder
            '..',                       # Go "up" to the main project folder
            'data'                      # Go "down" into the 'data' folder
        )
        
        surfaces_path = os.path.join(save_dir, 'iv_surfaces.npy')
        dates_path = os.path.join(save_dir, 'dates.npy')
        scalars_path = os.path.join(save_dir, 'cond_array.npy')

        # np.save(surfaces_path, final_surfaces)
        # np.save(dates_path, final_dates)
        # np.save(scalars_path, final_scalars)
        
        print("Successfully saved 3 files:")
        print(f"  - iv_surfaces.npy: {final_surfaces.shape}")
        print(f"  - dates.npy:         {final_dates.shape}")
        print(f"  - cond_array.npy:    {final_scalars.shape}")
    except Exception as e:
        print(f"\nAn error occurred during final alignment or saving: {e}")

if __name__ == "__main__":
    acquire_and_clean_data()
