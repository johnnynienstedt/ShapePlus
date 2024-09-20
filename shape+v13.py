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

# changes from v11:
    # back to my HAA
    # test all outcomes
    # added pitcher-level correlations
    # replaced AB% with FB% (since LD and PU are not predictable)
    
    
import pybaseball
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import get_ipython
from matplotlib.ticker import FuncFormatter
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

for col in drop_cols:
    try:
        all_pitch_data.drop(columns = [col], inplace=True)
    except KeyError:
        pass
        
clean_pitch_data = all_pitch_data.dropna(subset = necessary_cols).copy()
clean_pitch_data['pfx_x'] = np.round(clean_pitch_data['pfx_x']*12)
clean_pitch_data['pfx_z'] = np.round(clean_pitch_data['pfx_z']*12)

# select pitchers with at least 100 pitches thrown
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

# my calculations of VAA and HAA
zf = pitcher_pitch_data['plate_z']
delta_z = rz - zf
delta_y = ry - yf
phi_z = -np.arctan(delta_z/delta_y)*180/np.pi

xf = pitcher_pitch_data['plate_x']
delta_x = rx - xf
phi_x = -np.arctan(delta_x/delta_y)*180/np.pi

pitcher_pitch_data.insert(89, 'my_VAA', round(2*phi_z - theta_z0, 2))
pitcher_pitch_data.insert(91, 'my_HAA', round(2*phi_x - theta_x0, 2))

delta = []
for v in range(75,102):
    VAA = pitcher_pitch_data[
        (pitcher_pitch_data['release_speed'] > v - 0.5) &
        (pitcher_pitch_data['release_speed'] < v + 0.5)]['VAA'].mean()
    my_VAA = pitcher_pitch_data[
        (pitcher_pitch_data['release_speed'] > v - 0.5) &
        (pitcher_pitch_data['release_speed'] < v + 0.5)
    ]['my_VAA'].mean()
    delta.append(VAA - my_VAA)
    
model = np.poly1d(np.polyfit(np.arange(75,102), delta, 2))

velo_correction = model[0] + model[1]*velo + model[2]*velo**2
geom_VAA = 2*phi_z - theta_z0 + velo_correction
geom_HAA = 2*phi_x - theta_x0 + velo_correction

pitcher_pitch_data.drop(columns=['my_VAA', 'my_HAA'], inplace=True)
pitcher_pitch_data.insert(89, 'geom_VAA', round(geom_VAA, 2))
pitcher_pitch_data.insert(91, 'geom_HAA', round(geom_HAA, 2))

# total break angle
delta_theta_z = theta_z0 - theta_zf
delta_theta_x = theta_x0 - theta_xf
delta_theta = np.sqrt(delta_theta_z**2 + delta_theta_x**2)
pitcher_pitch_data['break_angle'] = round(delta_theta, 2)

# sharpness of break
eff_t = delta_y/velo
iVB = pitcher_pitch_data['pfx_z']
sharpness = np.abs(iVB/eff_t)
pitcher_pitch_data['sharpness'] = round(sharpness, 2)





'''
###############################################################################
############################# Assign Pitch Result #############################
###############################################################################
'''

outcomes = ['swstr', 'gb', 'fb']

# run value of a...
swstr_rv = 0.116
gb_rv = 0.058
fb_rv = -0.107

# assign binary values for outcomes of interest
swstr_types = ['swinging_strike_blocked', 'swinging_strike', 'foul_tip']
swing_tpyes = swstr_types + ['foul', 'hit_into_play']
strike_zones = np.arange(1,10)
pitcher_pitch_data['zone_value'] = pitcher_pitch_data['zone'].isin(strike_zones).astype(int)
pitcher_pitch_data['swing_value'] = pitcher_pitch_data['description'].isin(swing_tpyes).astype(int)
pitcher_pitch_data['swstr_value'] = pitcher_pitch_data['description'].isin(swstr_types).astype(int)
pitcher_pitch_data['bip_value'] = (pitcher_pitch_data['description'] == 'hit_into_play').astype(int)
pitcher_pitch_data['gb_value'] = (pitcher_pitch_data['bb_type'] == 'ground_ball').astype(int)
pitcher_pitch_data['fb_value'] = ((pitcher_pitch_data['description'] == 'hit_into_play') & 
                                  (pitcher_pitch_data['bb_type'] == 'fly_ball')).astype(int)





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
    platoon_percent = (df.stand == df.p_throws).mean() * 100
    classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                             (classified_pitch_data.player_name == pitcher), 'platoon_percent'] = platoon_percent
    
    
    # get sinkers and 4-seamers for pitch shape baseline
    ff = df[df.pitch_type == 'FF']
    si = df[df.pitch_type == 'SI']
    
    ff_baseline = (ff.release_speed.mean(), ff.pfx_x.mean(), ff.pfx_z.mean())
    si_baseline = (si.release_speed.mean(), si.pfx_x.mean(), si.pfx_z.mean())
    
    ffvel, ffh, ffv = ff_baseline if len(ff) >= 10 else (94, 5, 14)
    sivel, sih, siv = si_baseline if len(si) >= 10 else (93, 12, 9)
    
    # If either pitch type is missing, adjust the baselines accordingly
    if len(si) < 10 and len(ff) > 10:
        sivel, sih, siv = ffvel - 1, ffh + 5, ffv - 5
    if len(ff) < 10 and len(si) > 10:
        ffvel, ffh, ffv = sivel + 1, sih - 5, siv + 5
    
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

    
    
    pitch_type_group = df.groupby('pitch_type')
    for pitch_type, group in pitch_type_group:
        pitch_shape = np.array([group.pfx_x.mean(), group.pfx_z.mean(), group.release_speed.mean()])
        if platoon_percent == 0:
            sh_percent = 0
            oh_percent  = 100
        elif platoon_percent == 100:
            sh_percent = 100
            oh_percent  = 0
        else:
            sh_percent = (group.stand == group.p_throws).sum() / round(platoon_percent/100*n) * 100
            oh_percent = (group.stand != group.p_throws).sum() / round((100 - platoon_percent)/100*n) * 100
    
        if pitch_type == 'KN':
            pitch_name = 'Knuckleball'
        else:
            distances = np.linalg.norm(pitch_archetypes - pitch_shape, axis=1)
            min_index = np.argmin(distances)
            pitch_name = pitch_names[min_index]
            
            if pitch_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
                if pitch_name == 'Movement-Based Changeup' and sivel - pitch_shape[2] > 6:
                    pitch_name = 'Velo-Based Changeup'
                elif pitch_name == 'Velo-Based Changeup' and sivel - pitch_shape[2] <= 6:
                    pitch_name = 'Movement-Based Changeup'
        
        mask = (classified_pitch_data.game_year == year) & \
               (classified_pitch_data.player_name == pitcher) & \
               (classified_pitch_data.pitch_type == pitch_type)
        
        classified_pitch_data.loc[mask, ['true_pitch_type', 'sh_percent', 'oh_percent']] = [
            pitch_name, round(sh_percent, 1), round(oh_percent, 1)
        ]
    
        if pitch_name in ['Riding Fastball', 'Fastball']:
            # Update archetypes and names after identification
            pitch_archetypes = np.delete(pitch_archetypes, min_index, axis=0)
            pitch_names = np.delete(pitch_names, min_index, axis=0)
            

for pitcher in tqdm(pitcher_pitch_data.player_name.unique()):
    get_repertoire(pitcher, year='all')





'''
###############################################################################
############################ Create Outcome Models ############################
###############################################################################
'''

# columns of interest
x_cols = ['release_speed', 'release_extension', 'pfx_x', 'pfx_z', 'VAA', 'HAA',]
y_cols = ['swstr_value', 'gb_value', 'fb_value']
display_cols = ['player_name'] + x_cols + y_cols

# pitch names
pitch_names = np.array([
    'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
    'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
    'Velo-Based Changeup', 'Knuckleball'
])
    
pitch_data = classified_pitch_data.copy()

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
sh_all_results = sh_data.loc[sh_data.game_year == 2024, display_cols + ['sh_percent']].reset_index(drop=True)
oh_all_results = oh_data.loc[oh_data.game_year == 2024, display_cols + ['oh_percent']].reset_index(drop=True)
sh_all_results['true_pitch_type'] = sh_data.loc[sh_data.game_year == 2024, 'true_pitch_type'].values
oh_all_results['true_pitch_type'] = oh_data.loc[oh_data.game_year == 2024, 'true_pitch_type'].values

sh_data_2024 = sh_data[sh_data['game_year'] == 2024]
pitch_counts = sh_data_2024.groupby(['player_name', 'true_pitch_type']).size().reset_index(name='count')
sh_all_results = sh_all_results.merge(
    pitch_counts,
    on=['player_name', 'true_pitch_type'],
    how='left'
)

oh_data_2024 = oh_data[oh_data['game_year'] == 2024]
pitch_counts = oh_data_2024.groupby(['player_name', 'true_pitch_type']).size().reset_index(name='count')
oh_all_results = oh_all_results.merge(
    pitch_counts,
    on=['player_name', 'true_pitch_type'],
    how='left'
)

# dictionary for storing models
sh_models_dict = {}
oh_models_dict = {}

# loop over each outcome
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
                                 objective = 'binary:logistic', seed = '42', 
                                 verbosity = 0)
    # fit model
    xgb_model.fit(xtrain, ytrain)
    
    # store model in dictionary
    sh_models_dict[outcome] = xgb_model
    
    # use model to predict outcome likelihood for 2024 pitches
    sh_all_results[outcome] = pd.Series(xgb_model.predict(eval_X))
    
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
                                 objective = 'binary:logistic', seed = '42', 
                                 verbosity = 0)
    # fit model
    xgb_model.fit(xtrain, ytrain)
    
    # store model in dictionary
    oh_models_dict[outcome] = xgb_model
    
    # use model to predict outcome probabilities for 2024 pitches
    oh_all_results[outcome] = pd.Series(xgb_model.predict(eval_X))


# plt.bar(x_cols, sh_models_dict['swstr'].feature_importances_)



'''
###############################################################################
################################ Grade Pitches ################################
###############################################################################
'''

# compute averages for each pitcher's pitch types
sh_shape_grades = sh_all_results.copy().groupby(['player_name', 'true_pitch_type']).mean(numeric_only=False).reset_index()
sh_shape_grades[x_cols] = np.round(sh_shape_grades[x_cols], 1)
sh_shape_grades[['pfx_x', 'pfx_z']] = np.round(sh_shape_grades[['pfx_x', 'pfx_z']])

oh_shape_grades = oh_all_results.copy().groupby(['player_name', 'true_pitch_type']).mean(numeric_only=False).reset_index()
oh_shape_grades[x_cols] = np.round(oh_shape_grades[x_cols], 1)
oh_shape_grades[['pfx_x', 'pfx_z']] = np.round(oh_shape_grades[['pfx_x', 'pfx_z']])

sh_shape_grades.rename(columns={'sh_percent': 'percent'}, inplace=True)
oh_shape_grades.rename(columns={'oh_percent': 'percent'}, inplace=True)
sh_shape_grades.rename(columns={'true_pitch_type': 'pitch_type'}, inplace=True)
oh_shape_grades.rename(columns={'true_pitch_type': 'pitch_type'}, inplace=True)


# normalize to mean 100, stdev 10
all_shape_grades = pd.concat([sh_shape_grades, oh_shape_grades])
cutoff = len(sh_shape_grades)
swstr_rate = all_shape_grades.swstr
gb_rate = all_shape_grades.gb
fb_rate = all_shape_grades.fb
xRV = swstr_rv*swstr_rate + gb_rv*gb_rate + fb_rv*fb_rate
xRV_mean = xRV.mean()
xRV_std = xRV.std()
norm_grades = ((xRV - xRV_mean)/xRV_std + 10)*10

all_shape_grades.insert(2, 'Shape+', np.round(norm_grades))
sh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[:cutoff]))
oh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[cutoff:]))

def grade_pitch(shape):
    
    names = x_cols
    shape_df = pd.DataFrame([shape], columns = names)
    
    probabilities = pd.DataFrame()
    for outcome in outcomes:
        
        sh_model = sh_models_dict[outcome]
        oh_model = oh_models_dict[outcome]
        
        probabilities.loc[0, outcome] = sh_model.predict(shape_df)[0]
        probabilities.loc[1, outcome] = oh_model.predict(shape_df)[0]
    
    xRV = probabilities.swstr*swstr_rv + probabilities.gb*gb_rv + probabilities.fb*fb_rv
    
    grades = ((xRV - xRV_mean)/xRV_std + 10)*10
    
    print()
    print(shape_df.to_string(index=False))
    print()
    print('    Same hand Shape+:', round(grades[0]))
    print('Opposite hand Shape+:', round(grades[1]))




'''
###############################################################################
############################### Grade Pitchers ################################
###############################################################################
'''

# display repertoire and grades
def grade_repertoire(pitcher, verbose = True, backend = 'qt'):
    
    
    # get all pitches from this year
    df = classified_pitch_data.copy().query('player_name == @pitcher and game_year == 2024', engine='python')
    
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
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0.05)
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
        legend = ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(-0.2, 0.6), fontsize = 15)
        ax.add_artist(legend)
        
        # set title
        fig.suptitle('2024 Pitch Repertoire -- ' + pfirst + ' ' + plast, fontsize=20)
        ax.text(30, -20, 70, "Data courtesy Baseball Savant", color='k', fontsize = 12)
        ax.set_xlabel('HB', fontsize=15)
        ax.set_ylabel('iVB', fontsize=15)
        ax.set_zlabel('Velo', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
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
        repertoire = sh_repertoire.merge(oh_repertoire, how='outer', on='pitch_type', suffixes=('_RHB', '_LHB'))
    if hand == 'L':
        repertoire = oh_repertoire.merge(sh_repertoire, how='outer', on='pitch_type', suffixes=('_RHB', '_LHB'))
        
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
        
        sh_percent = classified_pitch_data.query('player_name == @pitcher and game_year == 2024 and true_pitch_type == @pitch', engine='python')['sh_percent'].iloc[0]
        oh_percent = classified_pitch_data.query('player_name == @pitcher and game_year == 2024 and true_pitch_type == @pitch', engine='python')['oh_percent'].iloc[0]

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
        RHB_percent = round(100 - df.platoon_percent.values[0], 1)
    
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
    print(pfirst, plast, '(' + hand + 'HP) - 2024')
    print()
    print(repertoire.to_string(index=False))
    print()
    print('Total Shape+:', total_shape)
    
    
grade_repertoire('Estrada, Jeremiah', verbose=True)

    
# calculate aggregate grades for pitchers
sh_shape_grades['weighted_grade'] = sh_shape_grades['Shape+'] * sh_shape_grades['percent']
sh_shape_grades['weighted_swstr'] = sh_shape_grades['swstr'] * sh_shape_grades['percent']
sh_shape_grades['weighted_gb'] = sh_shape_grades['gb'] * sh_shape_grades['percent']
sh_shape_grades['weighted_fb'] = sh_shape_grades['fb'] * sh_shape_grades['percent']

sh_pitcher_grades = sh_shape_grades.groupby('player_name').apply(
    lambda x: pd.Series({
        'aggregate_grade': round(x['weighted_grade'].sum() / x['percent'].sum(), 1),
        'aggregate_swstr': round(x['weighted_swstr'].sum() * 100 / x['percent'].sum(), 1),
        'aggregate_gb': round(x['weighted_gb'].sum() * 100 / x['percent'].sum(), 1),
        'aggregate_fb': round(x['weighted_fb'].sum() * 100 / x['percent'].sum(), 1)
    })).reset_index()

sh_pitcher_grades['platoon_percent'] = round(sh_pitcher_grades.apply(lambda x: classified_pitch_data.query('player_name == @x.player_name and game_year == 2024', engine='python')['platoon_percent'].iloc[0], axis = 1), 3)

sh_pitcher_grades = sh_pitcher_grades.sort_values(by='aggregate_grade', ascending=False)

oh_shape_grades['weighted_grade'] = oh_shape_grades['Shape+'] * oh_shape_grades['percent']
oh_shape_grades['weighted_swstr'] = oh_shape_grades['swstr'] * oh_shape_grades['percent']
oh_shape_grades['weighted_gb'] = oh_shape_grades['gb'] * oh_shape_grades['percent']
oh_shape_grades['weighted_fb'] = oh_shape_grades['fb'] * oh_shape_grades['percent']

oh_pitcher_grades = oh_shape_grades.groupby('player_name').apply(
    lambda x: pd.Series({
        'aggregate_grade': round(x['weighted_grade'].sum() / x['percent'].sum(), 1),
        'aggregate_swstr': round(x['weighted_swstr'].sum() * 100 / x['percent'].sum(), 1),
        'aggregate_gb': round(x['weighted_gb'].sum() * 100 / x['percent'].sum(), 1),
        'aggregate_fb': round(x['weighted_fb'].sum() * 100 / x['percent'].sum(), 1)
    })).reset_index()
oh_pitcher_grades = oh_pitcher_grades.sort_values(by='aggregate_grade', ascending=False)

sh_pitch_grades = sh_shape_grades.copy().groupby('pitch_type').mean(numeric_only = True)
sh_pitch_grades['Shape+'] = sh_pitch_grades['Shape+'].astype(int)
sh_pitch_grades = sh_pitch_grades.drop(columns = ['percent', 'weighted_grade'])

oh_pitch_grades = oh_shape_grades.copy().groupby('pitch_type').mean(numeric_only = True)
oh_pitch_grades['Shape+'] = oh_pitch_grades['Shape+'].astype(int)
oh_pitch_grades = oh_pitch_grades.drop(columns = ['percent', 'weighted_grade'])

# merge the DataFrames on player_name
merged_grades = pd.merge(
    sh_pitcher_grades, oh_pitcher_grades, on='player_name', how='outer', suffixes=('_sh', '_oh')
)

# fill NaNs with 0 for missing grades or platoon_percent
merged_grades = merged_grades.fillna(0)

# calculate total_grade as the sum of the products
merged_grades['Shape+'] = round((
    merged_grades['aggregate_grade_sh'] * merged_grades['platoon_percent'] +
    merged_grades['aggregate_grade_oh'] * (100 - merged_grades['platoon_percent']))/100)

# calculate estimated outcome rates
merged_grades['Predicted_SwStr%'] = round((
    merged_grades['aggregate_swstr_sh'] * merged_grades['platoon_percent'] +
    merged_grades['aggregate_swstr_oh'] * (100 - merged_grades['platoon_percent']))/100, 1)
merged_grades['Predicted_GB%'] = round((
    merged_grades['aggregate_gb_sh'] * merged_grades['platoon_percent'] +
    merged_grades['aggregate_gb_oh'] * (100 - merged_grades['platoon_percent']))/100, 1)
merged_grades['Predicted_FB%'] = round((
    merged_grades['aggregate_fb_sh'] * merged_grades['platoon_percent'] +
    merged_grades['aggregate_fb_oh'] * (100 - merged_grades['platoon_percent']))/100, 1)


# create final pitcher_grades DataFrame
pitcher_grades = merged_grades[['player_name', 'Shape+', 'Predicted_SwStr%', 'Predicted_GB%', 'Predicted_FB%']].copy()
pitcher_grades['player_name'] = pitcher_grades['player_name'].str.split(',').apply(lambda x: x[1].strip() + ' ' + x[0].strip())





'''
###############################################################################
########################## Pitch-Level Correlations ###########################
###############################################################################
'''
    
sh_shape_corr = sh_shape_grades[sh_shape_grades['count'] > 100]
oh_shape_corr = oh_shape_grades[oh_shape_grades['count'] > 100]


def pitcher_pitch_correlations(ind_colname, dep_colname, h='s'):
    
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'inline')
    
    if h == 's':
        x = sh_shape_corr[ind_colname]
        y = sh_shape_corr[dep_colname]
    
    else:
        x = oh_shape_corr[ind_colname]
        y = oh_shape_corr[dep_colname]
    
    name_dict = {
                'swstr': 'Predicted SwStr%',
                'swstr_value': 'Actual SwStr%',
                'gb': 'Predicted GB%',
                'gb_value': 'Actual GB%',
                'fb': 'Predicted FB%',
                'fb_value': 'Actual FB%',
                
                }
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()
    
    xname = name_dict[ind_colname]
    yname = name_dict[dep_colname]

    ax.scatter(x, y, s=3)
    ax.plot(x, m*x+b, '--k')
    ax.set_title(xname + ' vs. ' + yname)
    ax.set_xlabel(xname + ' (Pitcher Pitch Type Level)')
    ax.set_ylabel(yname)
    
    # ax.set_aspect('equal', adjustable='box')
    
    left = x.min()
    top = y.max()*0.9
    ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
    
    plt.show()
    
    
pitcher_pitch_correlations('swstr', 'swstr_value')
pitcher_pitch_correlations('gb', 'gb_value')
pitcher_pitch_correlations('fb', 'fb_value')
    
    
    
'''
###############################################################################
######################### Pitcher-Level Correlations ##########################
###############################################################################
'''
    
pitcher_results_24 = pybaseball.pitching_stats(2024, qual = 1, ind = 1)

# pitcher_results.insert(1, 'pitcher', pybaseball.playerid_reverse_lookup(pitcher_results.IDfg, key_type='fangraphs').key_mlbam)
# pitcher_results['pitcher'] = pitcher_results.groupby('Name')['pitcher'].transform(lambda x: x.ffill().bfill())

pitcher_results_24 = pd.merge(pitcher_grades, pitcher_results_24, left_on='player_name', right_on='Name', how='inner')


def pitcher_correlations(ind_colname, dep_colname, q=400):
        
    filtered_results = pitcher_results_24[pitcher_results_24.Pitches > q]
    
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'inline')
    
    x = filtered_results[ind_colname]
    y = filtered_results[dep_colname]

    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()

    ax.scatter(x, y, s=3)
    ax.plot(x, m*x+b, '--k')
    ax.set_title('2024 ' + ind_colname + ' vs. ' + dep_colname + ' (min. ' + str(q) + ' pitches)')
    ax.set_xlabel(ind_colname + ' (Pitcher Level)')
    ax.set_ylabel(dep_colname)
    
    # ax.set_aspect('equal', adjustable='box')
    
    left = x.min()
    top = y.max()*0.9
    ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
    
    plt.show()

    
pitcher_correlations('Predicted_SwStr%', 'SwStr%', q=400)
pitcher_correlations('Predicted_GB%', 'GB%', q=400)
pitcher_correlations('Predicted_FB%', 'FB%', q=400)

pitcher_correlations('Shape+', 'SIERA', q=1000)



'''
###############################################################################
########################## Pitcher-Level Stickiness ###########################
###############################################################################
'''

pitcher_results = pybaseball.pitching_stats(2021, 2024, qual = 1, ind = 1)
pitcher_results.sort_values(by = ['IDfg', 'Season'], inplace=True)
pitcher_results['Whiff%'] = pitcher_results['SwStr%']/pitcher_results['Swing%']
pitcher_results['Whiff% (sc)'] = pitcher_results['SwStr%']/pitcher_results['Swing% (sc)']


# Stickiness year over year
def yoy(colname, q=400):
    
    filtered_results = pitcher_results[pitcher_results.Pitches > q].reset_index()
    filtered_results.dropna(subset = [colname])

    rows = []
    for i in range(len(filtered_results) - 1):
        if filtered_results['IDfg'][i + 1] == filtered_results['IDfg'][i]:
            c1 = filtered_results[colname][i]
            c2 = filtered_results[colname][i + 1]
            c3 = filtered_results['Pitches'][i]
            rows.append({'Year1': c1,
                         'Year2': c2,
                         'N_P': c3})
            
            
    pairs = pd.DataFrame(rows)
    
    
    # make plot
    x = pairs.Year1
    y = pairs.Year2
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=3)
    plt.plot(x, m*x+b, '--k')
    ax.set_title(colname + " YOY (Pitcher Level, min. " + str(q) + " Pitches Per Season)")
    ax.set_xlabel('Year 1 ' + colname)
    ax.set_ylabel('Year 2 ' + colname)
        
    # text
    y_limits = ax.get_ylim()
    bot = y_limits[0] + 0.1*(y_limits[1] - y_limits[0])
    top = y_limits[1] - 0.1*(y_limits[1] - y_limits[0])
    x_limits = ax.get_xlim()
    left = x_limits[0] + 0.1*(x_limits[1] - x_limits[0])
    right = x_limits[1] - 0.1*(x_limits[1] - x_limits[0])
    
    if r > 0:
        ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)), ha='left', va='center', fontsize=10)
    else:
        ax.text(right, top, '$R^2$ = ' + str(round(r**2, 2)), ha='right', va='center', fontsize=10)
    
    ax.text((left + right)/2, bot, 'Data via Baseball Savant 2021-2024', ha='center', va='top', fontsize=8)
    
    # percent if necessary
    if '%' in colname:
        if (y.max() - y.min()) > 0.1:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.0f}%'))
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.0f}%'))
        else:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.1f}%'))
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.1f}%'))
    
    plt.show()

    return pairs

pairs = yoy('Whiff%', q=1000)



# next steps:

# def compare_pitchers(pitcher_list):
# run values from each outcome pls