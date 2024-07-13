import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import trange


def ConvHulls(home_xy: pd.DataFrame, 
              away_xy: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute the convex hulls of the home and away teams.
    INPUTS:
    - home_xy: pd.DataFrame with the coordinates of the home players
    - away_xy: pd.DataFrame with the coordinates of the away players
    OUTPUT:
    - hulls_df: pd.DataFrame with the convex hulls of the home and away teams
    """
    hulls_df=pd.DataFrame(columns=['Period','Frame','Time [s]','HomeHull','AwayHull'])
    for frame in trange(home_xy.shape[0],desc="Convex Hulls estimation:"):
        time=home_xy.iloc[frame,:]['Time [s]']
        period=home_xy.iloc[frame,:]['Period']
        home_data=home_xy.iloc[frame,:]
        away_data=away_xy.iloc[frame,:]
        home_data=home_data.dropna()
        away_data=away_data.dropna()
        ball=np.array(home_data[-2:])
        home_data= home_data[4:-2] #exclude both the goalkeeper and the ball
        away_data= away_data[4:-2] #exclude both the goalkeeper and the ball
        #--------------------------------------------------------------------
        # divide x and y
        home_data_x=home_data[home_data.index.str.contains('_x')]
        home_data_y=home_data[home_data.index.str.contains('_y')]
        away_data_x=away_data[away_data.index.str.contains('_x')]
        away_data_y=away_data[away_data.index.str.contains('_y')]
        #--------------------------------------------------------------------
        # Coordinates
        home_pts= np.array([[x,y] for x,y in zip(home_data_x,home_data_y)])
        away_pts= np.array([[x,y] for x,y in zip(away_data_x,away_data_y)])
        # Compute the convex hulls
        home_hull=ConvexHull(home_pts)
        away_hull=ConvexHull(away_pts)
        # Compute the area of the convex hulls
        home_area=home_hull.volume
        away_area=away_hull.volume
        
        hulls_df.loc[frame]=[period,frame,time,home_area,away_area]

    return hulls_df