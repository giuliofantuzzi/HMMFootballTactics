import numpy as np
import pandas as pd
import utils.Metrica_Viz as mviz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import glob
from PIL import Image
from IPython.display import Image as IPImage
from IPython.display import display
import os

def PlotPitch(home_xy: pd.DataFrame,
              away_xy: pd.DataFrame,
              frame: int,
              lag: int=0,
              fig_size: tuple = (12, 7),
              plotHulls: bool=True,
              plotAllPlayers: bool=True,
              title: str='Convex Hulls of Home and Away Teams') -> plt.Figure:
    """
    Function to plot the convex hulls of the home and away teams.
    INPUTS:
    - home_pts: np.array with the coordinates of the home players
    - away_pts: np.array with the coordinates of the away players
    - ball: np.array with the coordinates of the ball
    - fig_size: tuple with the size of the figure
    - plotHulls: boolean to plot the convex hulls
    - plotAllPlayers: boolean to plot all the players or only the vertices
    - title: string with the title of the plot
    OUTPUT:
    - fig: matplotlib figure
    """
    #--------------------------------------------------------------------------------------------------------
    # Retrieve the data for the frame
    
    home_data=home_xy.iloc[frame-lag,:] #-50 since the match start
    away_data=away_xy.iloc[frame-lag,:]
    home_data=home_data.dropna()
    away_data=away_data.dropna()
    ball=np.array(home_data[-2:])
    home_data= home_data[4:-2] #exclude both the goalkeeper and the ball
    away_data= away_data[4:-2] #exclude both the goalkeeper and the ball
    
    # divide x and y
    home_data_x=home_data[home_data.index.str.contains('_x')]
    home_data_y=home_data[home_data.index.str.contains('_y')]
    away_data_x=away_data[away_data.index.str.contains('_x')]
    away_data_y=away_data[away_data.index.str.contains('_y')]
    
    # Coordinates
    home_pts= np.array([[x,y] for x,y in zip(home_data_x,home_data_y)])
    away_pts= np.array([[x,y] for x,y in zip(away_data_x,away_data_y)])
    
    #--------------------------------------------------------------------------------------------------------
    # Call plot_pitch to get fig and ax
    fig, ax = mviz.plot_pitch()
    fig.set_size_inches(fig_size)
    #--------------------------------------------------------------------------------------------------------
    
    # Plotting the convex hulls
    if plotHulls:
        home_hull=ConvexHull(home_pts)
        away_hull=ConvexHull(away_pts)
        for simplex in home_hull.simplices:
            ax.plot(np.array(home_pts)[simplex, 0], np.array(home_pts)[simplex, 1], 'k-')
        for simplex in away_hull.simplices:
            ax.plot(np.array(away_pts)[simplex, 0], np.array(away_pts)[simplex, 1], 'k-')
        #--------------------------------------------------------------------------------------------------------
        # Plotting the vertices and filling the convex hulls
        home_vertices = np.array(home_pts)[home_hull.vertices]
        ax.fill(home_vertices[:, 0], home_vertices[:, 1], 'blue', alpha=0.45, edgecolor='black')
        away_vertices = np.array(away_pts)[away_hull.vertices]
        ax.fill(away_vertices[:, 0], away_vertices[:, 1], 'red', alpha=0.45, edgecolor='black')
    #--------------------------------------------------------------------------------------------------------
    if plotAllPlayers: # (Option 1) Scatter ALL the players
        ax.scatter(home_pts[:,0], home_pts[:,1],
                label='Home Team',
                color='blue', s=100, edgecolor='black', zorder=2)
        ax.scatter(away_pts[:,0], away_pts[:,1],
                label='Away Team',s=100, color='red', edgecolor='black', zorder=2)
    else:  # (Option 2) Scatter ONLY the vertices
        ax.scatter(home_vertices[:, 0], home_vertices[:, 1], label='Home Team',
                color='blue', s=100, edgecolor='black', zorder=2)
        ax.scatter(away_vertices[:, 0], away_vertices[:, 1], label='Away Team',
                s=100, color='red', edgecolor='black', zorder=2)
    #--------------------------------------------------------------------------------------------------------
    # Ball
    ax.scatter(ball[0],ball[1], label='Ball', 
               s=75, color='white', edgecolor='black', zorder=2)
    #--------------------------------------------------------------------------------------------------------
    # Adding legend and title
    ax.legend(loc='lower right', fontsize=12, facecolor='white', edgecolor='black', bbox_to_anchor=(.97, 0.05));
    ax.set_title(title, fontsize=15, fontweight='bold')
    #--------------------------------------------------------------------------------------------------------
    plt.close(fig) # Close the figure to avoid double plotting
    return fig

def PlotGIF(home_xy: pd.DataFrame, 
            away_xy: pd.DataFrame, 
            initial_frame: int=0,
            final_frame: int=100,
            lag: int=0,
            step: int=1,
            gifname=None,
            title=None) -> None:
    """
    Function to create and display a GIF with the convex hulls of the home and away teams.
    The function saves the images in the figs folder and the GIF in the gifs folder.
    INPUTS:
    - home_xy: pd.DataFrame with the coordinates of the home players
    - away_xy: pd.DataFrame with the coordinates of the away players
    - initial_frame: int with the initial frame to plot
    - final_frame: int with the final frame to plot
    OUTPUT:
    - GIF displayed in the notebook
    """
    # Eliminate the figs directory
    os.system("rm -rf figs")
    os.system("mkdir figs")
    
    # Plotting the convex hulls for all frames in the range
    for frame in range(initial_frame,final_frame+1,step):     
        curr_plot=PlotPitch(home_xy=home_xy,away_xy=away_xy,frame=frame,lag=lag,plotHulls=True,plotAllPlayers=True,title=title)
        #save it
        curr_plot.savefig(f'figs/convex_hulls_{frame}.png', dpi=300, bbox_inches='tight')

    # Generate the GIF

    # Load all the saved images
    image_files = sorted(glob.glob('figs/convex_hulls_*.png'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Create a list of images
    # Desired GIF dimensions
    gif_width = 1200
    gif_height = 700
    images = [Image.open(image).resize((gif_width, gif_height), Image.Resampling.LANCZOS) for image in image_files]

    # Save as a GIF
    images[0].save(gifname, save_all=True, append_images=images, duration=0.0001, loop=0)

    # Display the GIF
    display(IPImage(gifname))
    
def plotEPS_with_states(data : pd.DataFrame,class_colors : dict,home_goals:pd.DataFrame=None,away_goals: pd.DataFrame=None,home_shot: pd.DataFrame=None,away_shot: pd.DataFrame=None):
    """
    Args:
        data (pd.DataFrame): dataframe with the EPS
        home_goals (pd.DataFrame): dataframe with the home goals
        away_goals (pd.DataFrame): dataframe with the home goals
        home_shot (pd.DataFrame): dataframe with the home shots
        away_shot (pd.DataFrame): dataframe with the away shots
        class_colors (dict): dictionary with the colors for each state
    Returns:
        plt.Figure
    """
    n_states = len(data["State"].unique())
    assert len(class_colors)==n_states
    colors = data['State'].map(class_colors)
    fig, axs = plt.subplots(2, 1, figsize=(36, 12))
    # Home team's convex hull area
    axs[0].vlines(data["Time [s]"]/60, ymin=0, ymax=data["HomeHull"], color=colors, linewidth=0.4)
    axs[0].set_xlabel("Time [min]", fontsize=12,fontweight='normal');
    axs[0].set_ylabel("Convex Hull Area", fontsize=12, fontweight='normal');
    axs[0].set_title("Convex Hull Area over Time (Home team)", fontsize=15, fontweight='bold');
    # add legend
    for i, state in enumerate(list(range(n_states))):
        axs[0].plot([], [], color=class_colors[int(state)], label=f'State {int(state)}')
    axs[0].legend(title="State", title_fontsize='11', fontsize='11', loc='upper right');

    # Away team's convex hull area
    axs[1].vlines(data["Time [s]"]/60, ymin=0, ymax=data["AwayHull"],color=colors, linewidth=0.4)
    axs[1].set_xlabel("Time [min]", fontsize=12,fontweight='normal');
    axs[1].set_ylabel("Convex Hull Area", fontsize=12, fontweight='normal');
    axs[1].set_title("Convex Hull Area over Time (Away team)", fontsize=15, fontweight='bold');
    for i, state in enumerate(list(range(n_states))):
        axs[1].plot([], [], color=class_colors[int(state)], label=f'State {int(state)}')
    axs[1].legend(title="State", title_fontsize='11', fontsize='11', loc='upper right');
    
    if (home_shot is not None) and (away_shot is not None):
        # add vertical lines for home shots
        for t1,t2 in zip(home_shot["Start Time [s]"]/60,home_shot["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='navy', linestyle='--', linewidth=3) 
            axs[1].axvline(x=(t1+t2)/2, color='navy', linestyle='--', linewidth=3)
        # add vertical lines for away shots
        for t1,t2 in zip(away_shot["Start Time [s]"]/60,away_shot["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='red', linestyle='--', linewidth=3) 
            axs[1].axvline(x=(t1+t2)/2, color='red', linestyle='--', linewidth=3)
            
    if (home_goals is not None) and (away_goals is not None):    
    # add vertical lines for home goals
        for t1,t2 in zip(home_goals["Start Time [s]"]/60,home_goals["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='navy', linestyle='-', linewidth=3.5) 
            axs[1].axvline(x=(t1+t2)/2, color='navy', linestyle='-', linewidth=3.5)
        # add vertical lines for away goals
        for t1,t2 in zip(away_goals["Start Time [s]"]/60,away_goals["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='red', linestyle='-', linewidth=3.5) 
            axs[1].axvline(x=(t1+t2)/2, color='red', linestyle='-', linewidth=3.5)
    # Manage space between subplots
    plt.subplots_adjust(hspace=0.3)
    plt.close(fig)
    return fig


def plotEPS_with_states_single(home_data : pd.DataFrame, away_data: pd.DataFrame, 
                            class_colors : dict,
                            home_goals:pd.DataFrame=None,away_goals: pd.DataFrame=None,home_shot: pd.DataFrame=None,away_shot: pd.DataFrame=None):
    """
    Args:
        data (pd.DataFrame): dataframe with the EPS
        home_goals (pd.DataFrame): dataframe with the home goals
        away_goals (pd.DataFrame): dataframe with the home goals
        home_shot (pd.DataFrame): dataframe with the home shots
        away_shot (pd.DataFrame): dataframe with the away shots
        class_colors (dict): dictionary with the colors for each state
    Returns:
        plt.Figure
    """
    n_states = len(home_data["State"].unique())
    assert len(class_colors)==n_states
    colors_home = home_data['State'].map(class_colors)
    colors_away = away_data['State'].map(class_colors)
    fig, axs = plt.subplots(2, 1, figsize=(36, 12))
    # fig, axs = plt.subplots(figsize=(36, 6))
    # Home team's convex hull area
    axs[0].vlines(home_data["Time [s]"]/60, ymin=0, ymax=home_data["HomeHull"], color=colors_home, linewidth=0.4)
    axs[0].set_xlabel("Time [min]", fontsize=12,fontweight='normal');
    axs[0].set_ylabel("Convex Hull Area", fontsize=12, fontweight='normal');
    axs[0].set_title("Convex Hull Area over Time (Home team)", fontsize=15, fontweight='bold');
    # add legend
    for i, state in enumerate(list(range(n_states))):
        axs[0].plot([], [], color=class_colors[int(state)], label=f'State {int(state)}')
    axs[0].legend(title="State", title_fontsize='11', fontsize='11', loc='upper right');

    # Away team's convex hull area
    axs[1].vlines(away_data["Time [s]"]/60, ymin=0, ymax=away_data["AwayHull"],color=colors_away, linewidth=0.4)
    axs[1].set_xlabel("Time [min]", fontsize=12,fontweight='normal');
    axs[1].set_ylabel("Convex Hull Area", fontsize=12, fontweight='normal');
    axs[1].set_title("Convex Hull Area over Time (Away team)", fontsize=15, fontweight='bold');
    for i, state in enumerate(list(range(n_states))):
        axs[1].plot([], [], color=class_colors[int(state)], label=f'State {int(state)}')
    axs[1].legend(title="State", title_fontsize='11', fontsize='11', loc='upper right');
    
    if (home_shot is not None) and (away_shot is not None):
        # add vertical lines for home shots
        for t1,t2 in zip(home_shot["Start Time [s]"]/60,home_shot["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='navy', linestyle='--', linewidth=3) 
            axs[1].axvline(x=(t1+t2)/2, color='navy', linestyle='--', linewidth=3)
        # add vertical lines for away shots
        for t1,t2 in zip(away_shot["Start Time [s]"]/60,away_shot["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='red', linestyle='--', linewidth=3) 
            axs[1].axvline(x=(t1+t2)/2, color='red', linestyle='--', linewidth=3)
            
    if (home_goals is not None) and (away_goals is not None):    
    # add vertical lines for home goals
        for t1,t2 in zip(home_goals["Start Time [s]"]/60,home_goals["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='navy', linestyle='-', linewidth=3.5) 
            axs[1].axvline(x=(t1+t2)/2, color='navy', linestyle='-', linewidth=3.5)
        # add vertical lines for away goals
        for t1,t2 in zip(away_goals["Start Time [s]"]/60,away_goals["End Time [s]"]/60):
            axs[0].axvline(x=(t1+t2)/2, color='red', linestyle='-', linewidth=3.5) 
            axs[1].axvline(x=(t1+t2)/2, color='red', linestyle='-', linewidth=3.5)
    # Manage space between subplots
    plt.subplots_adjust(hspace=0.3)
    plt.close(fig)
    return fig    



def plotEPS_distribution(data, class_colors,title="EPS copula distribution (by state)"):
    """
    Args:
        data: pd.DataFrame
        class_colors: dictionary with the colors for each state
    Returns:
        plt.Figure
    """
    n_states = len(data["State"].unique())
    assert len(class_colors)==n_states
    
    fig, ax = plt.subplots()
    sns.kdeplot(
        data=data,
        x="HomeHull", 
        y="AwayHull", 
        hue="State", 
        palette=class_colors,
        ax=ax
    )
    ax.set_title(title)
    plt.close(fig)
    return fig

def plotEPS_hist(data,class_colors):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))  # Adjusted figsize for better visualization

    sns.histplot(x=data["HomeHull"]*100, hue=data["State"], kde=True, palette=class_colors, alpha=0.2, ax=ax[0])
    ax[0].set_xlabel("Convex Hull Area", fontsize=12, fontweight='normal')
    ax[0].set_ylabel("Count", fontsize=12, fontweight='normal')
    ax[0].set_title("Home Team Convex Hull Area by State", fontsize=15, fontweight='bold')

    sns.histplot(x=data["AwayHull"], hue=data["State"], kde=True, palette=class_colors, alpha=0.2, ax=ax[1])
    ax[1].set_xlabel("Convex Hull Area", fontsize=12, fontweight='normal')
    ax[1].set_ylabel("Count", fontsize=12, fontweight='normal')
    ax[1].set_title("Away Team Convex Hull Area by State", fontsize=15, fontweight='bold')

    plt.subplots_adjust(hspace=0.45)
    plt.close(fig)
    return fig