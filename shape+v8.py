#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:12:18 2024

@author: johnnynienstedt
"""

#
# Early attempts at Shape+
# Johnny Nienstedt 8/1/24
#

#
# The purpose of this script is to develop a metric which grades pitch shapes
# on a standardized scale. This metric will NOT include any information about
# other complimentary pitches, but will include pitcher-specific information
# such as extension and release height.
#
# Ideally, this metric will be used in conjunction with another metric which
# grades the interaction between pitches to grade a pitcher's arsenal. This
# could give insight into which pitchers are getting the most out of their
# pitches by pairing them well and which could be in need of a change to their
# repetiore.
#

# changes from v6:
    # no fouls

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm
from IPython import get_ipython
from sklearn.model_selection import train_test_split


'''
###############################################################################
############################# Import & Clean Data #############################
###############################################################################
'''

# import pitch data from 2021-2024
all_pitch_data = pd.read_csv('all_pitch_data.csv', index_col=False)

drop_cols = ['Unnamed: 0', 'level_0', 'index']
necessary_cols = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax',
                  'ay', 'az', 'release_pos_x', 'release_pos_y', 'release_pos_z', 
                  'release_extension']

clean_pitch_data = all_pitch_data.copy().drop(columns = drop_cols)
clean_pitch_data = clean_pitch_data.dropna(subset = necessary_cols)
clean_pitch_data['pfx_x'] = np.round(clean_pitch_data['pfx_x']*12)
clean_pitch_data['pfx_z'] = np.round(clean_pitch_data['pfx_z']*12)

# select pitchers with at least 1000 pitches thrown
pitcher_pitch_data = clean_pitch_data.groupby('pitcher').filter(lambda x: len(x) >= 100)

# flip axis for RHP so that +HB = arm side, -HB = glove side
mirror_cols = ['release_pos_x', 'plate_x', 'pfx_x', 'vx0', 'ax']
pitcher_pitch_data.loc[pitcher_pitch_data['p_throws'] == 'R', mirror_cols] = -pitcher_pitch_data.loc[pitcher_pitch_data['p_throws'] == 'R', mirror_cols]





'''
###############################################################################
######################## Calculate Secondary Parameters #######################
###############################################################################
'''

# get relevant primary parameters
vx0 = pitcher_pitch_data['vx0']
vy0 = pitcher_pitch_data['vy0']
vz0 = pitcher_pitch_data['vz0']
ax = pitcher_pitch_data['ax']
ay = pitcher_pitch_data['ay']
az = pitcher_pitch_data['az']
rx = pitcher_pitch_data['release_pos_x']
ry = pitcher_pitch_data['release_pos_y']
rz = pitcher_pitch_data['release_pos_z']
velo = pitcher_pitch_data['release_speed']
y0 = 50
yf = 17/12

# vertical and horizontal release angle
theta_z0 = -np.arctan(vz0/vy0)*180/np.pi
theta_x0 = -np.arctan(vx0/vy0)*180/np.pi
pitcher_pitch_data['release_angle_v'] = round(theta_z0, 2)
pitcher_pitch_data['release_angle_h'] = round(theta_x0, 2)

# vertical and horizontal approach angle
vyf = -np.sqrt(vy0**2- (2 * ay * (y0 - yf)))
t = (vyf - vy0)/ay
vzf = vz0 + (az*t)
vxf = vx0 + (ax*t)

theta_zf = -np.arctan(vzf/vyf)*180/np.pi
theta_xf = -np.arctan(vxf/vyf)*180/np.pi
pitcher_pitch_data['VAA'] = round(theta_zf, 2)
pitcher_pitch_data['HAA'] = round(theta_xf, 2)

# total break angle
delta_theta_z = theta_z0 - theta_zf
delta_theta_x = theta_x0 - theta_xf
delta_theta = np.sqrt(delta_theta_z**2 + delta_theta_x**2)
pitcher_pitch_data['break_angle'] = round(delta_theta, 2)

# sharpness of break
eff_t = (ry - yf)/velo
sharpness = delta_theta/eff_t
pitcher_pitch_data['sharpness'] = round(sharpness, 2)


# maybe introduce some sanity checks here? In terms of pitch break vs location,
# making sure everythin adds up.





'''
###############################################################################
############################ Evaluate Pitch Result ############################
###############################################################################
'''

# run value of a...
swstr_rv = 0.116
gb_rv = 0.058
ab_rv = -0.141

outcomes = ['swstr', 'gb', 'ab']

swstr_types = ['swinging_strike_blocked', 'swinging_strike', 'foul_tip']

def get_swstr_value(description):
    
    if description in swstr_types:
        return 1
        
    else:
        return 0
    
def get_gb_value(description, bb_type):
    
    if description == 'hit_into_play':
        
        if bb_type == 'ground_ball':
            return 1
        
        else:
            return 0
        
    else:
        return 0

def get_ab_value(description, bb_type):
    
    if description == 'hit_into_play':
        
        if bb_type == 'ground_ball':
            return 0
        
        else:
            return 1
    else:
        return 0
    
# set values for each pitch
pitcher_pitch_data['swstr_value'] = pitcher_pitch_data.apply(lambda x: get_swstr_value(x.description), axis = 1)
pitcher_pitch_data['gb_value'] = pitcher_pitch_data.apply(lambda x: get_gb_value(x.description, x.bb_type), axis = 1)
pitcher_pitch_data['ab_value'] = pitcher_pitch_data.apply(lambda x: get_ab_value(x.description, x.bb_type), axis = 1)






'''
###############################################################################
############################# Classify Pitch Types ############################
###############################################################################
'''

classified_pitch_data = pitcher_pitch_data.copy()

# function for determining repertoires
def get_repertoire(pitcher, year = 'all'):
    
    # select proper year(s)
    if year == 'all':
        years = pitcher_pitch_data[pitcher_pitch_data.player_name == pitcher].game_year.unique()
        for year in years:
            get_repertoire(pitcher, year)
        return
    else:
        df = pitcher_pitch_data[(pitcher_pitch_data.player_name == pitcher) & 
                                (pitcher_pitch_data.game_year == year)].copy().reset_index(drop=True)  
    
    # number of pitches thrown
    n = len(df)
    if n == 0:
        raise AttributeError('No data for this pitcher & year(s).')
    
    # percent thrown to same-handed batters
    classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                          (classified_pitch_data.player_name == pitcher), 'platoon_percent'] = sum(df.stand == df.p_throws)/n*100
    
    # get sinkers and 4-seamers for pitch shape baseline
    ff = df[df.pitch_type == 'FF']
    si = df[df.pitch_type == 'SI']
    
    # get baseline velo, HB, and iVB
    if len(ff) >= 10 and len(si) >= 10:
        ffvel = ff.release_speed.mean()
        ffh = ff.pfx_x.mean()
        ffv = ff.pfx_z.mean()
        
        sivel = si.release_speed.mean()
        sih = si.pfx_x.mean()
        siv = si.pfx_z.mean()
        
    elif len(ff) >= 10 and len(si) < 10:
        ffvel = ff.release_speed.mean()
        ffh = ff.pfx_x.mean()
        ffv = ff.pfx_z.mean()
        
        sivel = ffvel - 1
        sih = ffh + 5
        siv = ffv - 5
        
    elif len(ff) < 10 and len(si) >= 10:
        sivel = si.release_speed.mean()
        sih = si.pfx_x.mean()
        siv = si.pfx_z.mean()
        
        ffvel = sivel + 1
        ffh = sih - 5
        ffv = siv + 5
    
    else:
        ffvel = 94
        ffh = 5
        ffv = 14
        
        sivel = 93
        sih = 12
        siv = 9
    
    # pitch archetypes
    pitch_archetypes = np.array([
        [ffh, 18, ffvel],  # Riding Fastball
        [ffh, 10, ffvel],  # Fastball
        [sih, siv, sivel],  # Sinker
        [-3, 8, ffvel - 3],  # Cutter
        [-3, 0, ffvel - 9],  # Gyro Slider
        [-8, 0, ffvel - 11],  # Two-Plane Slider
        [-16, 1, ffvel - 14],  # Sweeper
        [-16, -6, ffvel - 15],  # Slurve
        [-8, -12, ffvel - 16],  # Curveball
        [-8, -12, ffvel - 22], # Slow Curve
        [sih, siv - 5, sivel - 4],  # Movement-Based Changeup
        [sih, siv - 5, sivel - 10]   # Velo-Based Changeup
    ])
     
    # pitch names
    pitch_names = np.array([
        'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
        'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
        'Velo-Based Changeup', 'Knuckleball'
    ])

    
    for pitch_type in df.pitch_type.unique():
        
        # retrieve all instances of this pitch
        this_pitch = df[df.pitch_type == pitch_type]
        
        # basic pitch shape characteristics
        HB = round(this_pitch.pfx_x.mean(), 1)
        iVB = round(this_pitch.pfx_z.mean(), 1)
        velo = round(this_pitch.release_speed.mean(), 1)
        pitch_shape = np.array([HB, iVB, velo])
        
        # usage of this pitch, by handedness
        sh_count = sum(df.stand == df.p_throws)
        oh_count = sum(df.stand != df.p_throws)
        
        if sh_count != 0:
            sh_percent = sum(this_pitch.stand == this_pitch.p_throws)/sh_count*100
        else:
            sh_percent = 0
        
        if oh_count != 0:
            oh_percent = sum(this_pitch.stand != this_pitch.p_throws)/oh_count*100
        else:
            oh_percent = 0
        
        
        if (oh_percent + sh_percent)/2 < 0.01:
            
            continue
        
        # handle knuckleballs
        if pitch_type == 'KN':
            
            classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                      (classified_pitch_data.player_name == pitcher) &
                                      (classified_pitch_data.pitch_type == pitch_type), 'true_pitch_type'] = 'Knuckleball'
            classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                      (classified_pitch_data.player_name == pitcher) &
                                      (classified_pitch_data.pitch_type == pitch_type), 'sh_percent'] = round(sh_percent, 1)
            classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                      (classified_pitch_data.player_name == pitcher) &
                                      (classified_pitch_data.pitch_type == pitch_type), 'oh_percent'] = round(oh_percent, 1)
            
            continue

        # calculate Euclidean distance
        distances = np.linalg.norm(pitch_archetypes - pitch_shape, axis=1)
        
        # find closest match
        min_index = np.argmin(distances)
        pitch = pitch_names[min_index]
        
        # more work on pesky changeups
        if pitch == 'Movement-Based Changeup':
            if sivel - velo > 6:
                try:
                    pitch = 'Velo-Based Changeup'
                    min_index = np.where(pitch_names == pitch)[0][0]
                except IndexError:
                    pass
        if pitch == 'Velo-Based Changeup':
            if sivel - velo <= 6:
                try:
                    pitch = 'Movement-Based Changeup'
                    min_index = np.where(pitch_names == pitch)[0][0]
                except IndexError:
                    pass
                    
        # make new columns
        classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                  (classified_pitch_data.player_name == pitcher) &
                                  (classified_pitch_data.pitch_type == pitch_type), 'true_pitch_type'] = pitch
        classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                  (classified_pitch_data.player_name == pitcher) &
                                  (classified_pitch_data.pitch_type == pitch_type), 'sh_percent'] = round(sh_percent, 1)
        classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                                  (classified_pitch_data.player_name == pitcher) &
                                  (classified_pitch_data.pitch_type == pitch_type), 'oh_percent'] = round(oh_percent, 1)
        
        if pitch == 'Riding Fastball':
            pitch_archetypes = np.delete(pitch_archetypes, 0, axis = 0)
            pitch_names = np.delete(pitch_names, 0, axis = 0)

        if pitch == 'Fastball':
            pitch_archetypes = np.delete(pitch_archetypes, 0, axis = 0)
            pitch_names = np.delete(pitch_names, 0, axis = 0)
    

for pitcher in tqdm(pitcher_pitch_data.player_name.unique()):
    get_repertoire(pitcher, year='all')





'''
###############################################################################
############################ Create Shape+ Models #############################
###############################################################################
'''

# xgboost!!!

# predictor columns
x_cols = ['release_speed', 'pfx_x', 'pfx_z', 'release_extension', 'VAA', 'HAA', 'sharpness']
y_cols = ['swstr_value', 'gb_value', 'ab_value']
display_cols = ['player_name'] + x_cols

# pitch names
pitch_names = np.array([
    'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
    'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
    'Velo-Based Changeup', 'Knuckleball'
])

sh_all_results = pd.DataFrame()
oh_all_results = pd.DataFrame()
for pitch in tqdm(pitch_names):
    
    n = sum(classified_pitch_data.true_pitch_type == pitch)
    
    if n == 0:
        continue
    
    # select this pitch
    pitch_data = classified_pitch_data.copy().loc[classified_pitch_data.true_pitch_type == pitch, :]
    
    # separate by handedness
    sh_data = pitch_data.loc[pitch_data.stand == pitch_data.p_throws, :]
    oh_data = pitch_data.loc[pitch_data.stand != pitch_data.p_throws, :]
    
    # use 2021-2023 for training data
    sh_X = sh_data.loc[sh_data.game_year < 2024, x_cols]
    oh_X = oh_data.loc[oh_data.game_year < 2024, x_cols]
    sh_Y = sh_data.loc[sh_data.game_year < 2024, y_cols]
    oh_Y = oh_data.loc[oh_data.game_year < 2024, y_cols]
    
    # use 2024 for evaluations
    sh_eval_X = sh_data.loc[sh_data.game_year == 2024, x_cols].reset_index(drop=True)
    oh_eval_X = oh_data.loc[oh_data.game_year == 2024, x_cols].reset_index(drop=True)
    
    # for display
    sh_pitch_results = sh_data.loc[sh_data.game_year == 2024, display_cols].reset_index(drop=True)
    oh_pitch_results = oh_data.loc[oh_data.game_year == 2024, display_cols].reset_index(drop=True)
    sh_pitch_results['pitch_type'] = [pitch] * len(sh_pitch_results)
    oh_pitch_results['pitch_type'] = [pitch] * len(oh_pitch_results)

    
    for outcome in outcomes:
        
        #
        # Same handedness
        #
        
        # select proper data
        y_col = outcome + '_value'
        Y = sh_Y.loc[:, y_col]
        X = sh_X
        eval_X = sh_eval_X
        
        # split into train/test sets
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)
        
        # set xgboost model
        xgb_model = xgb.XGBRegressor(learning_rate=0.21,max_depth = 4, 
                                     objective = 'reg:squarederror', seed = '42', 
                                     verbosity = 0)
        # fit model
        xgb_model.fit(xtrain, ytrain)
        
        # use model to predict outcome likelihood for 2024 pitches
        sh_pitch_results[outcome] = pd.Series(xgb_model.predict(eval_X))
        
        #
        # Opposite handedness
        #
        
        # select proper data
        y_col = outcome + '_value'
        Y = oh_Y.loc[:, y_col]
        X = oh_X
        eval_X = oh_eval_X
        
        # split into train/test sets
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)
        
        # set xgboost model
        xgb_model = xgb.XGBRegressor(learning_rate=0.21,max_depth = 4, 
                                     objective = 'reg:squarederror', seed = '42', 
                                     verbosity = 0)
        # fit model
        xgb_model.fit(xtrain, ytrain)
        
        # use model to predict outcome likelihood for 2024 pitches
        oh_pitch_results[outcome] = pd.Series(xgb_model.predict(eval_X))
        
    
    # # limit to pitchers who've thrown at least 10 of this pitch type
    # sh_pitch_results = sh_pitch_results.groupby('player_name').filter(lambda x: len(x) >= 10)
    # oh_pitch_results = oh_pitch_results.groupby('player_name').filter(lambda x: len(x) >= 10)
    
    # concatenate with main dataframe
    sh_all_results = pd.concat([sh_all_results, sh_pitch_results], ignore_index=True)
    oh_all_results = pd.concat([oh_all_results, oh_pitch_results], ignore_index=True)


# compute averages for each pitcher's pitch types
sh_shape_grades = sh_all_results.copy().groupby(['player_name', 'pitch_type']).mean(numeric_only=False).reset_index()
sh_shape_grades[['release_speed', 'release_extension', 'VAA', 'HAA', 'sharpness']] = np.round(sh_shape_grades[['release_speed', 'release_extension', 'VAA', 'HAA', 'sharpness']], 1)
sh_shape_grades[['pfx_x', 'pfx_z']] = np.round(sh_shape_grades[['pfx_x', 'pfx_z']])

oh_shape_grades = oh_all_results.copy().groupby(['player_name', 'pitch_type']).mean(numeric_only=False).reset_index()
oh_shape_grades[['release_speed', 'release_extension', 'VAA', 'HAA', 'sharpness']] = np.round(oh_shape_grades[['release_speed', 'release_extension', 'VAA', 'HAA', 'sharpness']], 1)
oh_shape_grades[['pfx_x', 'pfx_z']] = np.round(oh_shape_grades[['pfx_x', 'pfx_z']])

# add percent column
sh_shape_grades['percent'] = sh_shape_grades.apply(lambda x: classified_pitch_data.query('player_name == @x.player_name and game_year == 2024 and true_pitch_type == @x.pitch_type')['sh_percent'].iloc[0], axis = 1)
oh_shape_grades['percent'] = oh_shape_grades.apply(lambda x: classified_pitch_data.query('player_name == @x.player_name and game_year == 2024 and true_pitch_type == @x.pitch_type')['oh_percent'].iloc[0], axis = 1)

# normalize to mean 100, stdev 10
swstr_rate = sh_shape_grades.swstr
gb_rate = sh_shape_grades.gb
ab_rate = sh_shape_grades.ab
xRV = swstr_rv*swstr_rate + gb_rv*gb_rate + ab_rv*ab_rate
norm_grades = ((xRV - xRV.mean())/xRV.std() + 10)*10
sh_shape_grades.insert(2, 'Shape+', np.round(norm_grades))

swstr_rate = oh_shape_grades.swstr
gb_rate = oh_shape_grades.gb
ab_rate = oh_shape_grades.ab
xRV = swstr_rv*swstr_rate + gb_rv*gb_rate + ab_rv*ab_rate
norm_grades = ((xRV - xRV.mean())/xRV.std() + 10)*10
oh_shape_grades.insert(2, 'Shape+', np.round(norm_grades))

# display repertoire and grades
def grade_repertoire(pitcher, verbose = True, backend = 'qt'):
    
    
    # get all pitches from this year
    df = classified_pitch_data.copy().query('player_name == @pitcher and game_year == 2024')
    
    # pitcher first and last name
    pfirst = pitcher.split(', ')[1]
    plast = pitcher.split(', ')[0]
    
    # pitcher handedness
    hand = df.p_throws.values[0]
    
    # number of pitches thrown
    n = len(df)
    if n == 0:
        raise AttributeError('No data for ' + pfirst + ' ' + plast + ' in 2024. Make sure you include accents.')
    

    
    
    #
    # plot all pitches by type
    #

    
    if verbose:
        
        # select plotting method
        if backend == 'qt':
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'qt')
        else:
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'inline')
        
        # pitches + corresponding colors
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
            'Velo-Based Changeup', 'Knuckleball'
        ])
        
        colors = np.array(['red', 'tomato', 'darkorange', 'sienna', 'xkcd:sunflower yellow', 
                           'yellowgreen', 'limegreen', 'lightseagreen', 
                           'cornflowerblue', 'mediumpurple', 'darkgoldenrod',
                           'goldenrod', 'gray'])
        
        colordict = dict(zip(pitch_names, colors))
    
        # add colors
        df['color'] = df['true_pitch_type'].map(colordict)
        df = df.dropna(subset = 'color')
    
        # redo this for plotting
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based\n Changeup', 
            'Velo-Based\n Changeup', 'Knuckleball'
        ])
        
        colordict = dict(zip(pitch_names, colors))
    
        # sort by usage
        df = df.sort_values(by='sh_percent', ascending=False)
        
        # all pitch shapes
        HB = list(df.pfx_x)
        iVB = list(df.pfx_z)
        velo = list(df.release_speed)
        
        # make plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(HB, iVB, velo, c=df.color)
        
        # make legend handles and labels
        pitch_arsenal = list(df.true_pitch_type.unique())
        try:
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in pitch_arsenal]
        except KeyError:
            for i in range(len(pitch_arsenal)):
                if pitch_arsenal[i] in ('Movement-Based Changeup', 'Velo-Based Changeup'):
                    pitch_arsenal[i] = pitch_arsenal[i].split()[0] + '\n ' + pitch_arsenal[i].split()[1]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in pitch_arsenal]
        
        # sort handles and labels
        legend_labels = pitch_arsenal
        sorted_handles_labels = [(handles[i], legend_labels[i]) for i in range(len(legend_labels))]
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        
        # make legend
        legend = ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(-0.35, 0.65))
        ax.add_artist(legend)
        
        # plot archeypes
        # ax.scatter(pitch_archetypes[:,0], pitch_archetypes[:,1], pitch_archetypes[:,2], color = 'red')
        
        
        # set title
        ax.set_title('2024 Pitch Repertoire -- ' + pfirst + ' ' + plast)
        
        ax.text(35, -20, 70, "Data courtesy Baseball Savant", color='k', fontsize = 7.5)
        ax.set_xlabel('HB')
        ax.set_ylabel('iVB')
        ax.set_zlabel('Velo')
        ax.set_xlim((-20,20))
        ax.set_ylim((-20,20))
        ax.set_zlim((70,100))
        
        plt.show()
        
        
    #
    # print repertoire grades
    #
    
    
    # repertoire grades
    sh_repertoire = sh_shape_grades.copy().loc[sh_shape_grades.player_name == pitcher, ['pitch_type', 'Shape+']].reset_index(drop=True)
    oh_repertoire = oh_shape_grades.copy().loc[oh_shape_grades.player_name == pitcher, ['pitch_type', 'Shape+']].reset_index(drop=True)
    if hand == 'R':
        repertoire = sh_repertoire.merge(oh_repertoire, how='left', on='pitch_type', suffixes=('_RHB', '_LHB'))
    if hand == 'L':
        repertoire = oh_repertoire.merge(sh_repertoire, how='left', on='pitch_type', suffixes=('_RHB', '_LHB'))
        
    # make extra columns
    repertoire.insert(1, 'HB', [None]*len(repertoire))
    repertoire.insert(2, 'iVB', [None]*len(repertoire))
    repertoire.insert(3, 'Velo', [None]*len(repertoire))
    repertoire.insert(4, 'Usage_RHB', [None]*len(repertoire))
    repertoire.insert(6, 'Usage_LHB', [None]*len(repertoire))


    # get usage and chapes
    for pitch in repertoire.pitch_type:
        HB = df.query('true_pitch_type == @pitch')['pfx_x'].mean()
        iVB = df.query('true_pitch_type == @pitch')['pfx_z'].mean()
        velo = df.query('true_pitch_type == @pitch')['release_speed'].mean()
        
        repertoire.loc[repertoire.pitch_type == pitch, 'HB'] = round(HB)
        repertoire.loc[repertoire.pitch_type == pitch, 'iVB'] = round(iVB)
        repertoire.loc[repertoire.pitch_type == pitch, 'Velo'] = round(velo, 1)
        
        sh_percent = classified_pitch_data.query('player_name == @pitcher and game_year == 2024 and true_pitch_type == @pitch')['sh_percent'].iloc[0]
        oh_percent = classified_pitch_data.query('player_name == @pitcher and game_year == 2024 and true_pitch_type == @pitch')['oh_percent'].iloc[0]

        if hand == 'R':
            repertoire.loc[repertoire.pitch_type == pitch, 'Usage_RHB'] = sh_percent
            repertoire.loc[repertoire.pitch_type == pitch, 'Usage_LHB'] = oh_percent
        if hand == 'L':
            repertoire.loc[repertoire.pitch_type == pitch, 'Usage_RHB'] = oh_percent
            repertoire.loc[repertoire.pitch_type == pitch, 'Usage_LHB'] = sh_percent

    # sort by more prominent usage
    if hand == 'R':
        RHB_percent = round(df.platoon_percent.values[0], 1)
    else:
        RHB_percent = round(1 - df.platoon_percent.values[0], 1)
    
    if RHB_percent > 0.5:
        repertoire = repertoire.sort_values(by = 'Usage_RHB', ascending=False).reset_index(drop=True)
    else:
        repertoire = repertoire.sort_values(by = 'Usage_LHB', ascending=False).reset_index(drop=True)
    
    # fill NAs for unused pitches
    repertoire = repertoire.fillna(0)
    
    # add totals row
    RHB_shape = round(sum(repertoire['Shape+_RHB']*repertoire.Usage_RHB/100))
    LHB_shape = round(sum(repertoire['Shape+_LHB']*repertoire.Usage_LHB/100))
    total_shape = round((RHB_shape*RHB_percent + LHB_shape*(100 - RHB_percent))/100)
    total_row = ['Total', '', '', '', RHB_percent, RHB_shape, 100 - RHB_percent, LHB_shape]
    repertoire.loc[len(repertoire)] =  total_row
    
    # for readability 
    repertoire = repertoire.rename(columns={'pitch_type': 'Pitch Type'})
    repertoire['Shape+_RHB'] = repertoire['Shape+_RHB'].astype(int)
    repertoire['Shape+_LHB'] = repertoire['Shape+_LHB'].astype(int)
    
    repertoire['Shape+_RHB'] = repertoire['Shape+_RHB'].replace({0: '--'})
    repertoire['Shape+_LHB'] = repertoire['Shape+_LHB'].replace({0: '--'})


    print()
    print(pfirst, plast, '- 2024')
    print()
    print(repertoire.to_string(index=False))
    print()
    print('Total Shape+:', total_shape)
    
    
# calculate aggregate grades for pitchers
sh_shape_grades['weighted_grade'] = sh_shape_grades['Shape+'] * sh_shape_grades['percent']
sh_pitcher_grades = sh_shape_grades.groupby('player_name').apply(
    lambda x: pd.Series({
        'aggregate_grade': round(x['weighted_grade'].sum() / x['percent'].sum(), 1)
    })).reset_index()
sh_pitcher_grades['platoon_percent'] = round(sh_pitcher_grades.apply(lambda x: classified_pitch_data.query('player_name == @x.player_name and game_year == 2024')['platoon_percent'].iloc[0], axis = 1), 3)

sh_pitcher_grades = sh_pitcher_grades.sort_values(by='aggregate_grade', ascending=False)

oh_shape_grades['weighted_grade'] = oh_shape_grades['Shape+'] * oh_shape_grades['percent']
oh_pitcher_grades = oh_shape_grades.groupby('player_name').apply(
    lambda x: pd.Series({
        'aggregate_grade': round(x['weighted_grade'].sum() / x['percent'].sum(), 1)
    })).reset_index()
oh_pitcher_grades['platoon_percent'] = round(100 - oh_pitcher_grades.apply(lambda x: classified_pitch_data.query('player_name == @x.player_name and game_year == 2024')['platoon_percent'].iloc[0], axis = 1), 3)

oh_pitcher_grades = oh_pitcher_grades.sort_values(by='aggregate_grade', ascending=False)

sh_pitch_grades = sh_shape_grades.copy().groupby('pitch_type').mean(numeric_only = True)
sh_pitch_grades['Shape+'] = sh_pitch_grades['Shape+'].astype(int)
sh_pitch_grades = sh_pitch_grades.drop(columns = ['percent', 'weighted_grade'])

oh_pitch_grades = oh_shape_grades.copy().groupby('pitch_type').mean(numeric_only = True)
oh_pitch_grades['Shape+'] = oh_pitch_grades['Shape+'].astype(int)
oh_pitch_grades = oh_pitch_grades.drop(columns = ['percent', 'weighted_grade'])

# Merge the DataFrames on player_name, allowing for players to appear in one or both DataFrames
merged_grades = pd.merge(
    sh_pitcher_grades, oh_pitcher_grades, on='player_name', how='outer', suffixes=('_sh', '_oh')
)

# Fill NaNs with 0 for missing grades or platoon_percent, as they contribute nothing to the total grade
merged_grades = merged_grades.fillna(0)

# Calculate total_grade as the sum of the products
merged_grades['total_grade'] = round((
    merged_grades['aggregate_grade_sh'] * merged_grades['platoon_percent_sh'] +
    merged_grades['aggregate_grade_oh'] * merged_grades['platoon_percent_oh']
)/100)

# Create the final pitcher_grades DataFrame with the required columns
pitcher_grades = merged_grades[['player_name', 'total_grade']]


grade_repertoire('Hoeing, Bryan', verbose=True)

# def compare_pitchers(pitcher_list):

# def grade_pitch(pitch_type,shape):
    
    
# def correlations(ind, dep):
    
