# %%
import os, tqdm, json, pickle, gc, zipfile, itertools, time
import pandas as pd
import numpy as np
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from multiprocessing import Pool
import catboost as cb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import optuna
from optuna.samplers import TPESampler
from tqdm.contrib.concurrent import process_map  
import seaborn as sns
import matplotlib.pyplot as plt
import shap 
from sklearn.model_selection import KFold
from nancorrmp.nancorrmp import NaNCorrMp
# from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
from datetime import datetime
from optuna.samplers import TPESampler

class CsgoOutcomePredictor():

    def __init__(self):
        pass    
    
    def get_game_collection(self, PATH_TO_DIR):
        
        """
        Описание: коллекционирование респонсов парсера
        Параметры: PATH_TO_DIR - путь до директории с респонсами
        """

        L_FILENAMES = os.listdir(PATH_TO_DIR)
        L_COLLECTION = []
        for fnm in tqdm.tqdm(L_FILENAMES):
            try:
                pth = os.path.join(PATH_TO_DIR, fnm)
                with open(pth, 'r') as f:
                    d_rsp = json.load(f)
                L_COLLECTION.append(d_rsp)
            except:
                pass
        idx_ordered = np.argsort([d_game['id'] for d_game in L_COLLECTION])[::-1]
        L_COLLECTION = np.array(L_COLLECTION)[idx_ordered].tolist()
        return L_COLLECTION

    def add_global_info(self, d_game):
        
        # информация об игре        
        d = {}

        d['id'] = d_game['id']
        d['match_id'] = d_game['match_id']
        d['match_type'] = d_game['match']['match_type']
        d['number_of_games'] = d_game['match']['number_of_games']
        d['date'] = parser.parse(d_game['begin_at'])
        d['map_id'] = d_game['map']['id']
        d['league_id'] = d_game['match']['league']['id']
        d['serie_id'] = d_game['match']['serie']['id']
        d['tournament_id'] = d_game['match']['tournament']['id']
        d['serie_tier'] = d_game['match']['serie']['tier']

        return d

    def add_profile(self, d_game):
            
        # идентификаторы актуальных карт
        l_map2use = [1, 2, 6, 7, 8, 20, 31]
        # ключи со статистикой игрока
        l_stat_keys = [
            'adr', 'assists', 'deaths', 'first_kills_diff', 'flash_assists', 
            'headshots', 'k_d_diff', 'kast', 'kills', 'rating'
        ]

        # информация об игре
        d_info = self.add_global_info(d_game)
        
        if d_info['map_id'] in l_map2use:  
            
            # проверка валидности раундов
            d_r1 = d_game['rounds'][0]
            total = len(d_game['rounds'])
            if (d_r1['round']==1)&(total>=16):
                
                # информация о раундах
                df_rounds = pd.DataFrame.from_records(d_game['rounds'])
                start_ct_id =d_r1['ct']   
                winner_id = df_rounds['winner_team'].value_counts().idxmax()
                maxround = df_rounds['round'].max()
                d_h1_win_count = df_rounds.query('round<=15')['winner_team'].value_counts().to_dict()
                d_h2_win_count = df_rounds.query('round>15')['winner_team'].value_counts().to_dict()
                d_h1_outcome_count = df_rounds.query('round<=15')['outcome'].value_counts().to_dict()
                d_h2_outcome_count = df_rounds.query('round>15')['outcome'].value_counts().to_dict()        

                L = []
                counter = 0
                # информация об игроках
                for p in d_game['players']:
                    counter+=1

                    d = {}
                    d.update(d_info)

                    # идентификатор игрока
                    d['player_id'] = p['player']['id']
                    # идентификатор команды
                    d['team_id'] = p['team']['id']
                    # идентификатор оппонента
                    d['opponent_id'] = p['opponent']['id']

                    # национальность игрока
                    d['player_nationality']  = p['player']['nationality']
                    # дата рождения игрока
                    d['player_birthday']  = p['player']['birthday']
                    # страна команды
                    d['team_location']  = p['team']['location']

                    # сторона начала
                    d['start_ct']= 1 if start_ct_id==d['team_id'] else 0
                    # победа
                    d['win'] = 1 if winner_id==d['team_id'] else 0
                    # все раундов в игре
                    d['maxround'] = maxround

                    # число выигранных раундов в 1-ой половине игры
                    try:
                        d['h1_win_count'] = d_h1_win_count[d['team_id']]
                    except:
                        d['h1_win_count'] = 0 
                    # число выигранных раундов во 2-ой половине игры
                    try:
                        d['h2_win_count'] = d_h2_win_count[d['team_id']]
                    except:
                        d['h2_win_count'] = 0 
                    # исходы раундов в 1-ой половине игры
                    for k, v in d_h1_outcome_count.items():
                        d[f'h1_outcome_{k}_count'] = v
                    # исходы раундов во 2-ой половине игры
                    for k, v in d_h2_outcome_count.items():
                        d[f'h2_outcome_{k}_count'] = v            

                    # статистика игрока
                    d.update({k:p[k] if pd.notnull(p[k]) else 0.0 for k in l_stat_keys})
                    d.update({f'{k}_per_round':p[k]/maxround if pd.notnull(p[k]) else 0.0 for k in l_stat_keys})

                    L.append(d)
                if counter==10:
                    return L
                else:
                    return None
            else:
                return None

    def get_profiles(self, L_COLLECTION):

        """
        Описание: профайлинг игроков в играх
        Параметры: L_COLLECTION- коллекция респонсов
        """        
            
        # информация об игре
        L_GLOBAL_KEYS = [
            'id', 'match_id', 'match_type', 'number_of_games',
            'date', 'year', 'month', 'day', 'weekday', 'hour',
            'map_id',
            'league_id', 'serie_id', 'tournament_id', 'serie_tier',
            'start_ct'
        ]
        # ключи для агрегирования
        L_AGG_KEYS = [    
            
            'h1_outcome_defused_count', 'h1_outcome_eliminated_count',
            'h1_outcome_exploded_count', 'h1_outcome_timeout_count',
            'h1_win_count', 'h2_outcome_defused_count',
            'h2_outcome_eliminated_count', 'h2_outcome_exploded_count',
            'h2_outcome_timeout_count', 'h2_win_count',

            'win', 'maxround',

            'adr', 'assists', 'deaths', 'first_kills_diff', 'flash_assists', 'headshots',
            'k_d_diff', 'kast', 'kills', 'rating', 
            'adr_per_round', 'assists_per_round', 'deaths_per_round', 'first_kills_diff_per_round', 'flash_assists_per_round', 'headshots_per_round',
            'k_d_diff_per_round', 'kast_per_round', 'kills_per_round','rating_per_round'
        ]
        # ключи для группировки
        L_GROUP_KEYS = [
            'team_id', 'opponent_id', 'team_location', 'lineup'
        ]

        # профайлинг игрока
        L_player_profile = []
        for d_game in tqdm.tqdm(L_COLLECTION):
            try:
                L_player_profile.extend(self.add_profile(d_game))        
            except:
                pass
        df_player_profile = pd.DataFrame.from_records(L_player_profile)
        del L_player_profile
        gc.collect()

        L_dict = []
        for (game_id, team_id), subdf in tqdm.tqdm(df_player_profile.groupby(['id', 'team_id'])):
            n_players = subdf.shape[0]
            if n_players==5:
                subdf_c = subdf.copy()
                lineup = '-'.join(subdf['player_id'].sort_values().astype(str))
                subdf_c['lineup'] = lineup
                L_dict.extend(subdf_c.to_dict('records'))
        del df_player_profile
        gc.collect()
        df_player_profile = pd.DataFrame.from_records(L_dict).sort_values('date')
        del L_dict
        gc.collect()

        date = df_player_profile['date']
        df_player_profile['year'] = date.dt.year
        df_player_profile['month'] = date.dt.month
        df_player_profile['day'] = date.dt.day
        df_player_profile['weekday'] = date.dt.weekday
        df_player_profile['hour'] = date.dt.hour
        df_player_profile[['serie_tier', 'team_location']] = df_player_profile[['serie_tier', 'team_location']].fillna('default')    

        # профайлинг команды
        L_team_profile = []
        for (game_id, team_id), subdf in tqdm.tqdm(df_player_profile.groupby(['id', 'team_id'])):    
            d = subdf[L_GLOBAL_KEYS+L_GROUP_KEYS].iloc[0].to_dict()    
            d.update(subdf[L_AGG_KEYS].mean().to_dict())
            L_team_profile.append(d)
        df_team_profile = pd.DataFrame.from_records(L_team_profile)
        del L_team_profile
        gc.collect()

        df_player_profile_c = df_player_profile.apply(self.reduce_mem_usage)
        del df_player_profile 
        df_team_profile_c = df_team_profile.apply(self.reduce_mem_usage)
        del df_team_profile 

        return df_player_profile_c, df_team_profile_c

    def add_info4game(self, game_id):  

        L_GAMEINFO_KEYS = [
            'id',
            'number_of_games',
            'year','month', 'day', 'weekday', 'hour',
            'map_id',
            'league_id', 'serie_id', 'tournament_id', 
            'serie_tier'
        ]
        
        df_game = self.df_team_profile.query('id==@game_id')

        d_fs4gm = {}
        d_fs4gm.update(df_game[L_GAMEINFO_KEYS].iloc[0].to_dict())

        d_team_id2start_ct = dict(zip(df_game['team_id'], df_game['start_ct']))
        d_team_id2opponent_id = dict(zip(df_game['team_id'], df_game['opponent_id']))
        d_team_id2lineup = dict(zip(df_game['team_id'], df_game['lineup']))
        d_team_id2loc = dict(zip(df_game['team_id'], df_game['team_location']))

        df_game = self.df_player_profile.query('id==@game_id')

        for team_id, subdf in df_game.groupby('team_id'):

            prefix = 'start_ct' if d_team_id2start_ct[team_id]==1 else 'start_t'

            d_fs4gm[f'{prefix}__team_id'] = team_id    
            d_fs4gm[f'{prefix}__team_lineup'] = d_team_id2lineup[team_id]
            d_fs4gm[f'{prefix}__team_location'] = d_team_id2loc[team_id]
            
            subdf = subdf.sort_values('player_id')    
            L_p_id = subdf['player_id'].values    
            d_player_id2nat = dict(zip(subdf['player_id'], subdf['player_nationality']))
            ser_bd = subdf['player_birthday'].astype('datetime64')
            ser_bd_y = ser_bd.dt.year
            ser_bd_m = ser_bd.dt.month
            ser_bd_d = ser_bd.dt.day

            for i, p_id in enumerate(L_p_id):
                d_fs4gm[f'{prefix}__player{i+1}_id'] = p_id
                d_fs4gm[f'{prefix}__player{i+1}_nationality'] = d_player_id2nat[p_id]
                d_fs4gm[f'{prefix}__player{i+1}_birthday_year'] = ser_bd_y.iloc[i]
                d_fs4gm[f'{prefix}__player{i+1}_birthday_month'] = ser_bd_m.iloc[i]
                d_fs4gm[f'{prefix}__player{i+1}_birthday_day'] = ser_bd_d.iloc[i]  
        return d_fs4gm

    def add_features__gameinfo(self, PATH_TO_GAMEINFO_FEATURES):        

        ls = os.listdir(PATH_TO_GAMEINFO_FEATURES)
        L_GAME_IDXS = np.unique(self.df_team_profile['id'])
        try:
            set_in = set([int(x.split('.')[0]) for x in ls])
        except:
            set_in = set()
        set_all = set(L_GAME_IDXS)
        set_new = set_all-set_in
        L_GAME_IDXS = list(set_new)[::-1]

        for game_id in tqdm.tqdm(L_GAME_IDXS):
            try:    
                d_fs4gm = self.add_info4game(game_id)
                pth = os.path.join(PATH_TO_GAMEINFO_FEATURES, '{}.pickle'.format(game_id))
                with open(pth, 'wb') as f:
                    pickle.dump(d_fs4gm, f)
                del d_fs4gm
            except:
                pass

    def add_features__team4game(self, game_id):  

        L_GROUP_KEYS = [        
            'number_of_games',
            'year','month', 'day', 'weekday', 'hour',
            'serie_tier'
        ]
        L_FILTER_KEYS = [
            'league_id', 'serie_id', 'tournament_id'
        ]

        # ключи для агрегирования
        L_AGG_KEYS = [  

            'maxround', 'win', 
            
            'h1_outcome_defused_count', 'h1_outcome_eliminated_count',
            'h1_outcome_exploded_count', 'h1_outcome_timeout_count',
            'h1_win_count', 'h2_outcome_defused_count',
            'h2_outcome_eliminated_count', 'h2_outcome_exploded_count',
            'h2_outcome_timeout_count', 'h2_win_count',    

            'adr', 'first_kills_diff', 'k_d_diff', 'kast','rating', 
            'assists_per_round', 'deaths_per_round',
            'flash_assists_per_round', 'headshots_per_round',
            'kills_per_round'
            
        ]

        L_BY_KEYS = [
            'number_of_games',
            'year','month', 'day', 'weekday', 'hour',
            'serie_tier'
        ]
        
        df_game = self.df_team_profile.query('id==@game_id')

        date = df_game['date'].iloc[0]
        map_id = df_game['map_id'].iloc[0]
        league_id = df_game['league_id'].iloc[0]
        serie_id = df_game['serie_id'].iloc[0]
        tournament_id = df_game['tournament_id'].iloc[0]
        d_filter = dict(zip(['league_id', 'serie_id', 'tournament_id'], [league_id, serie_id, tournament_id]))

        d_fs4gm = {'id':game_id}    

        d_team_id2start_ct = dict(zip(df_game['team_id'], df_game['start_ct']))
        d_team_id2opponent_id = dict(zip(df_game['team_id'], df_game['opponent_id']))
        d_team_id2lineup = dict(zip(df_game['team_id'], df_game['lineup']))
        d_team_id2loc = dict(zip(df_game['team_id'], df_game['team_location']))
        

        for team_id, start_ct in d_team_id2start_ct.items():

            opponent_id = d_team_id2opponent_id[team_id]
            lineup = d_team_id2lineup[team_id]

            prefix = 'start_ct' if start_ct==1 else 'start_t'

            df_history = self.df_team_profile.query('(date<@date)&(team_id==@team_id)')
            df_history_on_map_with_start = df_history.query('(map_id==@map_id)&(start_ct==@start_ct)')        
            df_history_with_lineup = df_history.query('lineup==@lineup')
            df_history_on_map_with_start_and_lineup = df_history.query('(map_id==@map_id)&(start_ct==@start_ct)&(lineup==@lineup)')
            df_history_pair = df_history.query('opponent_id==@opponent_id')
            df_history_on_map_with_start_and_pair = df_history.query('(map_id==@map_id)&(start_ct==@start_ct)&(opponent_id==@opponent_id)')

            L_DF = [
                df_history, df_history_on_map_with_start, 
                df_history_with_lineup, df_history_on_map_with_start_and_lineup,
                df_history_pair, df_history_on_map_with_start_and_pair
            ]
            L_SUFFIX = [
                'all_map_all_start', 'current_map_current_start', 
                'all_map_all_start__lineup', 'current_map_current_start__lineup',
                'all_map_all_start__pair', 'current_map_current_start__pair',
            ]

            for filter_key, filter_value in d_filter.items():
                for suffix, df in zip(['all_map_all_start', 'current_map_current_start'],
                                    [df_history, df_history_on_map_with_start, ]):
                    L_SUFFIX.append(filter_key)
                    L_DF.append(df[df[filter_key]==filter_value])

            d_dicts4team = dict(zip(L_SUFFIX, L_DF))
            del L_SUFFIX, L_DF

            
            for suffix, subdf in d_dicts4team.items():                       
                for key in L_AGG_KEYS:
                    values = subdf[key].values
                    d_fs4gm[f'{prefix}__team__{suffix}__{key}__mean'] = np.mean(values)
                    d_fs4gm[f'{prefix}__team__{suffix}__{key}__sum'] = np.sum(values)
                    for by_key in L_BY_KEYS:
                        for by_value, subsubdf in subdf.groupby(by_key):
                            values = subsubdf[key].values
                            try:
                                d_fs4gm[f'{prefix}__team__{suffix}__{by_key}_{int(by_value)}__{key}__mean'] = np.mean(values)
                                d_fs4gm[f'{prefix}__team__{suffix}__{by_key}_{int(by_value)}__{key}__sum'] = np.sum(values)
                            except:
                                d_fs4gm[f'{prefix}__team__{suffix}__{by_key}_{by_value}__{key}__mean'] = np.mean(values)
                                d_fs4gm[f'{prefix}__team__{suffix}__{by_key}_{by_value}__{key}__sum'] = np.sum(values)
            del d_dicts4team

        return d_fs4gm   

    def add_features__team(self, PATH_TO_FEATURES_TEAM): 
        ls = os.listdir(PATH_TO_FEATURES_TEAM)
        L_GAME_IDXS = np.unique(self.df_team_profile['id'])
        try:
            set_in = set([int(x.split('.')[0]) for x in ls])
        except:
            set_in = set()
        set_all = set(L_GAME_IDXS)
        set_new = set_all-set_in
        L_GAME_IDXS = list(set_new)[::-1]

        for game_id in tqdm.tqdm(L_GAME_IDXS):
            try:    
                d_fs4gm = self.add_features__team4game(game_id)
                pth = os.path.join(PATH_TO_FEATURES_TEAM, '{}.pickle'.format(game_id))
                with open(pth, 'wb') as f:
                    pickle.dump(d_fs4gm, f)
                del d_fs4gm
            except:
                pass   

    def add_features__player4game(self, game_id):  

        L_GROUP_KEYS = [        
            'number_of_games',
            'year','month', 'day', 'weekday', 'hour',
            'serie_tier'
        ]
        L_FILTER_KEYS = [
            'league_id', 'serie_id', 'tournament_id'
        ]

        # ключи для агрегирования
        L_AGG_KEYS = [  
            
            'adr', 'first_kills_diff', 'k_d_diff', 'kast', 'rating', 
            'assists_per_round', 'deaths_per_round',
            'flash_assists_per_round', 'headshots_per_round',
            'kills_per_round'
            
        ]

        L_BY_KEYS = [            
            'year','month', 'day', 'weekday', 'hour'
            
        ]
        
        df_game = self.df_player_profile.query('id==@game_id')

        date = df_game['date'].iloc[0]
        map_id = df_game['map_id'].iloc[0]
        league_id = df_game['league_id'].iloc[0]
        serie_id = df_game['serie_id'].iloc[0]
        tournament_id = df_game['tournament_id'].iloc[0]
        d_filter = dict(zip(['league_id', 'serie_id', 'tournament_id'], [league_id, serie_id, tournament_id]))

        d_fs4gm = {'id':game_id}    

        d_team_id2start_ct = dict(zip(df_game['team_id'], df_game['start_ct']))   

        for team_id, start_ct in d_team_id2start_ct.items():        

            prefix = 'start_ct' if start_ct==1 else 'start_t'

            L_p_id = np.unique(df_game.query('team_id==@team_id')['player_id'])

            for i, p_id in enumerate(L_p_id):

                df_in_team_history = self.df_player_profile.query('(date<@date)&(player_id==@p_id)&(team_id==@team_id)')
                df_in_team_history_on_map_with_start = df_in_team_history.query('(map_id==@map_id)&(start_ct==@start_ct)')  
                df_not_in_team_history = self.df_player_profile.query('(date<@date)&(player_id==@p_id)&(team_id!=@team_id)')
                df_not_in_team_history_on_map_with_start = df_in_team_history.query('(map_id==@map_id)&(start_ct==@start_ct)') 

                L_DF = [
                    df_in_team_history, df_in_team_history_on_map_with_start, 
                    df_not_in_team_history, df_not_in_team_history_on_map_with_start                
                ]
                L_SUFFIX = [
                    f'player{i+1}__in_team__all_map_all_start', f'player{i+1}__in_team__current_map_current_start', 
                    f'player{i+1}__not_in_team__all_map_all_start', f'player{i+1}__not_in_team__current_map_current_start', 
                ]        

                d_dicts4player = dict(zip(L_SUFFIX, L_DF))
                del L_SUFFIX, L_DF

            
                for suffix, subdf in d_dicts4player.items():                       
                    for key in L_AGG_KEYS:
                        values = subdf[key].values
                        d_fs4gm[f'{prefix}__{suffix}__{key}__mean'] = np.mean(values)
                        d_fs4gm[f'{prefix}__{suffix}__{key}__sum'] = np.sum(values)
                        for by_key in L_BY_KEYS:
                            for by_value, subsubdf in subdf.groupby(by_key):
                                values = subsubdf[key].values
                                try:
                                    d_fs4gm[f'{prefix}__{suffix}__{by_key}_{int(by_value)}__{key}__mean'] = np.mean(values)
                                    d_fs4gm[f'{prefix}__{suffix}__{by_key}_{int(by_value)}__{key}__sum'] = np.sum(values)
                                except:
                                    d_fs4gm[f'{prefix}__{suffix}__{by_key}_{by_value}__{key}__mean'] = np.mean(values)
                                    d_fs4gm[f'{prefix}__{suffix}__{by_key}_{by_value}__{key}__sum'] = np.sum(values)
                del d_dicts4player

        return d_fs4gm

    def add_features__player(self, PATH_TO_FEATURES_PLAYER): 
        ls = os.listdir(PATH_TO_FEATURES_PLAYER)
        L_GAME_IDXS = np.unique(self.df_player_profile['id'])
        try:
            set_in = set([int(x.split('.')[0]) for x in ls])
        except:
            set_in = set()
        set_all = set(L_GAME_IDXS)
        set_new = set_all-set_in
        L_GAME_IDXS = list(set_new)[::-1]

        for game_id in tqdm.tqdm(L_GAME_IDXS):
            try:    
                d_fs4gm = self.add_features__player4game(game_id)
                pth = os.path.join(PATH_TO_FEATURES_PLAYER, '{}.pickle'.format(game_id))
                with open(pth, 'wb') as f:
                    pickle.dump(d_fs4gm, f)
                del d_fs4gm
            except:
                pass  

    def reduce_mem_usage(self, series):
        try:
            col_type = series.dtype

            if col_type != object:
                c_min = series.min()
                c_max = series.max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        series = series.astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        series = series.astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        series = series.astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        series = series.astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        series = series.astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        series = series.astype(np.float32)
                    else:
                        series = series.astype(np.float64)
            else:
                pass 
        except:
            pass
        
        return series 

    def build_features(self, PATH_TO_FEATURES_GAMEINFO, PATH_TO_FEATURES_TEAM, PATH_TO_FEATURES_PLAYER):

        """
        Сборка признаков
        """   

        # все файлы с признаками
        set_gameinfo= set(os.listdir(PATH_TO_FEATURES_GAMEINFO))
        set_team= set(os.listdir(PATH_TO_FEATURES_TEAM))
        set_player= set(os.listdir(PATH_TO_FEATURES_PLAYER))
        l_all_files = np.array(list(set.intersection(*[set_gameinfo, set_team, set_player])))
        l_all_files = l_all_files[np.argsort([int(x.split('.')[0]) for x in l_all_files])]

        # размер батча
        batch_size = 100
        n = np.int32(np.ceil(len(l_all_files) / batch_size))
        l_batches = np.array_split(l_all_files, n)

        # сборка
        df_features = pd.DataFrame()
        for batch in tqdm.tqdm(l_batches[-250:]):
            
            l = []
            for fnm in batch:
                D = {}
                for pth2dir in [PATH_TO_FEATURES_GAMEINFO, PATH_TO_FEATURES_TEAM, PATH_TO_FEATURES_PLAYER]:
                    pth = os.path.join(pth2dir, fnm)
                    with open(pth, 'rb') as f:
                        d = pickle.load(f)
                    D.update(d)
                    del d
                l.append(D)
                del D
            
            df = pd.DataFrame.from_records(l).apply(self.reduce_mem_usage)
            del l
            df_features = df_features.append(df)
            del df
            gc.collect()
        
        return df_features

    def build_targets(self, L_GAME_IDXS):
    
        """
        Сборка челевых переменных (победа, тотал м/б, число выигранных раундов в 1/2 половинах за обе стороны)
        """    
        df_targets = pd.DataFrame()
        for d_rsp in tqdm.tqdm(self.L_COLLECTION):  

            try:
                
                game_id = d_rsp['id']
                if game_id in L_GAME_IDXS:
                    ###########################################################################    
                    df_rounds = pd.DataFrame.from_records(d_rsp['rounds'])

                    maxround = df_rounds['round'].max()
                    start_ct_id = df_rounds.query('round==1')['ct'].iloc[0]
                    start_t_id = df_rounds.query('round==1')['terrorists'].iloc[0]
                    df_h1 = df_rounds.query('round<=15')
                    df_h2 = df_rounds.query('round>15')
                    d_h1_win_count = df_h1['winner_team'].value_counts().to_dict()
                    d_h2_win_count = df_h2['winner_team'].value_counts().to_dict()
                    d_h1h2_win_count = df_rounds['winner_team'].value_counts().to_dict()
                    winner_id = df_rounds['winner_team'].value_counts().idxmax()
                    

                    #############################################################################

                    d_targets4game = {'id':game_id}
                    
                    d_targets4game['start_ct__win'] = int(winner_id==start_ct_id)

                    for i in range(16, 31):

                        d_targets4game[f'total__b__{i}'] = int(maxround>=i)
                        d_targets4game[f'total__m__{i}'] = int(maxround<=i)

                    for i in range(1, 16):

                        d_targets4game[f'h1__start_ct_win__b__{i}'] = int(d_h1_win_count[start_ct_id]>=i)
                        d_targets4game[f'h1__start_ct_win__m__{i}'] = int(d_h1_win_count[start_ct_id]<=i)    
                        d_targets4game[f'h1__start_t_win__b__{i}'] = int(d_h1_win_count[start_t_id]>=i)
                        d_targets4game[f'h1__start_t_win__m__{i}'] = int(d_h1_win_count[start_t_id]<=i)

                        d_targets4game[f'h2__start_ct_win__b__{i}'] = int(d_h2_win_count[start_ct_id]>=i)
                        d_targets4game[f'h2__start_ct_win__m__{i}'] = int(d_h2_win_count[start_ct_id]<=i)    
                        d_targets4game[f'h2__start_t_win__b__{i}'] = int(d_h1_win_count[start_t_id]>=i)
                        d_targets4game[f'h2__start_t_win__m__{i}'] = int(d_h1_win_count[start_t_id]<=i)

                        d_targets4game[f'h1h2__start_ct_win__b__{i}'] = int(d_h1h2_win_count[start_ct_id]>=i)
                        d_targets4game[f'h1h2__start_ct_win__m__{i}'] = int(d_h1h2_win_count[start_ct_id]<=i)
                        d_targets4game[f'h1h2__start_t_win__b__{i}'] = int(d_h1h2_win_count[start_t_id]>=i)
                        d_targets4game[f'h1h2__start_t_win__m__{i}'] = int(d_h1h2_win_count[start_t_id]<=i)                     

                    df_targets = df_targets.append(d_targets4game, ignore_index = True)

            except:
                pass 
        df_targets['id'] = df_targets['id'].astype(int)
        
        return df_targets

    def prepare_data(self, df_targets, df_features):

        df_targets = df_targets.set_index('id').astype(int)
        df_features = df_features.set_index('id')
        games2use= np.intersect1d(df_features.index, df_targets.index)

        X = df_features.loc[games2use]
        del df_features
        gc.collect()
        L_CAT_FEATURES = [
            'number_of_games', 'year', 'month', 'day', 'weekday', 'hour',
            'map_id', 'league_id', 'serie_id', 'tournament_id', 'serie_tier',
            'start_t__team_id', 'start_t__team_lineup', 'start_t__team_location',
            'start_t__player1_id', 'start_t__player1_nationality',
            'start_t__player1_birthday_year', 'start_t__player1_birthday_month',
            'start_t__player1_birthday_day', 'start_t__player2_id',
            'start_t__player2_nationality', 'start_t__player2_birthday_year',
            'start_t__player2_birthday_month', 'start_t__player2_birthday_day',
            'start_t__player3_id', 'start_t__player3_nationality',
            'start_t__player3_birthday_year', 'start_t__player3_birthday_month',
            'start_t__player3_birthday_day', 'start_t__player4_id',
            'start_t__player4_nationality', 'start_t__player4_birthday_year',
            'start_t__player4_birthday_month', 'start_t__player4_birthday_day',
            'start_t__player5_id', 'start_t__player5_nationality',
            'start_t__player5_birthday_year', 'start_t__player5_birthday_month',
            'start_t__player5_birthday_day', 'start_ct__team_id',
            'start_ct__team_lineup', 'start_ct__team_location',
            'start_ct__player1_id', 'start_ct__player1_nationality',
            'start_ct__player1_birthday_year', 'start_ct__player1_birthday_month',
            'start_ct__player1_birthday_day', 'start_ct__player2_id',
            'start_ct__player2_nationality', 'start_ct__player2_birthday_year',
            'start_ct__player2_birthday_month', 'start_ct__player2_birthday_day',
            'start_ct__player3_id', 'start_ct__player3_nationality',
            'start_ct__player3_birthday_year', 'start_ct__player3_birthday_month',
            'start_ct__player3_birthday_day', 'start_ct__player4_id',
            'start_ct__player4_nationality', 'start_ct__player4_birthday_year',
            'start_ct__player4_birthday_month', 'start_ct__player4_birthday_day',
            'start_ct__player5_id', 'start_ct__player5_nationality',
            'start_ct__player5_birthday_year', 'start_ct__player5_birthday_month',
            'start_ct__player5_birthday_day'
        ]

        for key in L_CAT_FEATURES:
            try:
                X[key] = X[key].fillna(-9999).astype(int).astype('category')
            except:
                X[key] = X[key].fillna('default').astype('category')

        L_NUM_FEATURES = X.drop(L_CAT_FEATURES, 1).columns
        X[L_NUM_FEATURES] = X[L_NUM_FEATURES].fillna(-9999)
        
        Y = df_targets.loc[games2use]
        del df_targets
        gc.collect()    

        # X_obj = X.select_dtypes('category').astype('object')
        # L_obj_keys = X_obj.columns
        # for cmb in itertools.combinations(L_obj_keys, 2):
        #     cmb= list(cmb)
        #     new_key = '-'.join([str(x) for x in cmb])    
        #     X[new_key] = X_obj[cmb].astype('str').apply(lambda x: '-'.join(x), axis = 1).astype('category')
        # del X_obj
        # gc.collect()

        return X, Y  

    def update_dataset(self, X, Y, PATH_TO_DATASET):

        try:
            dataset = pd.read_pickle(PATH_TO_DATASET)   
            i = dataset['iter'].max()
            assert (X.index==Y.index).all()
            new_dataset = pd.concat([X.add_prefix('FEATURE_'), Y.add_prefix('TARGET_')], 1)
            del X, Y
            gc.collect()            
            new_dataset['update_at'] = datetime.now()
            new_dataset['iter'] = i+1
            dataset = pd.concat([dataset, new_dataset])
            dataset.to_pickle(PATH_TO_DATASET)
        except:
            assert (X.index==Y.index).all()
            new_dataset = pd.concat([X.add_prefix('FEATURE_'), Y.add_prefix('TARGET_')], 1)
            del X, Y
            gc.collect() 
            new_dataset['update_at'] = datetime.now()
            new_dataset['iter'] = 1
            new_dataset.to_pickle(PATH_TO_DATASET)  

    def run_ml_pipeline(self, PATH_TO_DATASET, target_key):

        def objective(trial):

            param = {
                "objective": trial.suggest_categorical("objective", ["Logloss"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", .5, 1),
                "depth": trial.suggest_int("depth", 3, 11),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli"]
                )
            }

            if param["bootstrap_type"] == "Bayesian":
                param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif param["bootstrap_type"] == "Bernoulli":
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            param.update(CONST_PARAMS)
            param['cat_features'] = np.where(X_train.dtypes=='category')[0]
            param['use_best_model'] = True
            params['random_state'] = SEED

            model = cb.CatBoostClassifier(**param)
            model.fit(X_train, y_train,  eval_set=(X_hold, y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)

            score = roc_auc_score(y_hold, model.predict_proba(X_hold)[:, 1])    
            
            return score
        ###################################################################################################    

        CONST_PARAMS= {
            'iterations':1000,
            'loss_function':'Logloss',    
            'verbose':0,
            }
        SEED=13
        N_PERM_ITER = 20
        TEST_SIZE, HOLD_SIZE = .1, .2
        EARLY_STOPPING_ROUNDS = 100
        
        dataset = pd.read_pickle(PATH_TO_DATASET)
        print('> dataset shape: {}'.format(dataset.shape))
        
        X = dataset.loc[:, dataset.columns.str.contains('FEATURE')]
        y = dataset[target_key].astype(int)
        del dataset
        gc.collect()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, shuffle = False)
        X_train, X_hold, y_train, y_hold = train_test_split(X_train, y_train, test_size = HOLD_SIZE, shuffle = False)
        del X, y
        gc.collect() 
        print('----------------------------------------------------------------------------------\n')

        print('> subspace feature selection ...')
        i = 0
        while True:    

            i+=1
            n_features = X_train.shape[1]
            print('\t> iter#{}. n_features: {}'.format(i, n_features))

            n_games = X_train.shape[0]
            batch_size = 5000        
            n_batches = np.int32(np.ceil(n_features/batch_size))
            L_feature_batch = np.array_split(X_train.columns, n_batches)

            if len(L_feature_batch)==1:
                break
            else:
                L_feat2use = []
                for batch in tqdm.tqdm(L_feature_batch):

                    try:
                        X_batch_train, X_batch_hold = X_train[batch], X_hold[batch]

                        params = CONST_PARAMS.copy()
                        params['cat_features'] = np.where(X_batch_train.dtypes=='category')[0]
                        params['use_best_model'] = True
                        params['random_state'] = SEED

                        model = cb.CatBoostClassifier(**params)   
                        model.fit(X_batch_train, y_train,  eval_set=(X_batch_hold, y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)

                        mask = model.feature_importances_>0            
                        L_feat2use.extend(X_batch_train.columns[mask].tolist())
                        del X_batch_train, X_batch_hold
                        gc.collect()  

                    except:
                        pass 

                X_train_c, X_hold_c = X_train[L_feat2use], X_hold[L_feat2use]
                del X_train, X_hold
                X_train, X_hold = X_train_c, X_hold_c
                del X_train_c, X_hold_c
                gc.collect()
        print('----------------------------------------------------------------------------------\n')   

        print('> recursive feature selection ...')
        i=0
        while True:
            
            i+=1
            n_features = X_train.shape[1]
            print('\t> iter#{}. n_features: {}'.format(i, n_features))

            params = CONST_PARAMS.copy()
            params['cat_features'] = np.where(X_train.dtypes=='category')[0]
            params['use_best_model'] = True
            params['random_state'] = SEED

            model = cb.CatBoostClassifier(**params)   
            model.fit(X_train, y_train,  eval_set=(X_hold, y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            
            mask = model.feature_importances_>0     
            if np.all(mask):
                break
            else:
                X_train_c, X_hold_c = X_train.loc[:, mask], X_hold.loc[:, mask]
                del X_train, X_hold
                X_train, X_hold = X_train_c, X_hold_c
                del X_train_c, X_hold_c
                gc.collect()
        print('----------------------------------------------------------------------------------\n')   

        print('> recursive feature selection with permutation importances...')
        i=0
        while True:

            i+=1
            n_features = X_train.shape[1]
            print('\t> iter#{}. n_features: {}'.format(i, n_features))

            study = optuna.create_study(
                sampler=TPESampler(),    
                direction="maximize"
                )
            study.optimize(objective, n_trials=100, timeout=60*10)

            best_params_before = {}
            best_params_before.update(CONST_PARAMS)
            best_params_before['cat_features'] = np.where(X_train.dtypes=='category')[0]
            best_params_before['use_best_model'] = True
            best_params_before['random_state'] = SEED
            best_params_before.update(study.best_trial.params)

            model = cb.CatBoostClassifier(**best_params_before)
            model.fit(X_train, y_train,  eval_set=(X_hold, y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            ho_score_before = roc_auc_score(y_hold, model.predict_proba(X_hold)[:, 1])

            z_perm_imp = np.zeros((X_train.shape[1], ))
            for _ in tqdm.tqdm(range(N_PERM_ITER)):

                model.fit(X_train, y_train,  eval_set=(X_hold, y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                
                imp = permutation_importance(
                        model,
                        X_hold, y_hold,
                        scoring = 'roc_auc',
                        n_repeats=1,
                        random_state = SEED+_,
                        n_jobs=-1
                        )['importances_mean'].flatten()
                        
                z_perm_imp += imp/N_PERM_ITER

            mask = z_perm_imp>0
            best_params_after = best_params_before.copy()
            best_params_after['cat_features'] = np.where(X_train.loc[:, mask].dtypes=='category')[0]
            model = cb.CatBoostClassifier(**best_params_after)
            model.fit(X_train.loc[:, mask], y_train,  eval_set=(X_hold.loc[:, mask], y_hold), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            ho_score_after = roc_auc_score(y_hold, model.predict_proba(X_hold.loc[:, mask])[:, 1])

            if ho_score_after>ho_score_before:
                X_train, X_hold = X_train.loc[:, mask], X_hold.loc[:, mask]
                X_test = X_test[X_train.columns] 
                best_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  
                best_features = X_train.columns
                best_params = best_params_after
            else:               
                break
        print('----------------------------------------------------------------------------------\n')                   

        return {'target':'_'.join(target_key.split('_')[1:]),
                'metric':best_score,
                'features':best_features,
                'params':best_params}


    def fit(self, PATH_TO_RESPONSES, PATH_TO_FEATURES_GAMEINFO, PATH_TO_FEATURES_TEAM, PATH_TO_FEATURES_PLAYER, PATH_TO_DATASET):

        time.sleep(1)
        print('> collecting responses ...')
        # коллекция респонсов
        self.L_COLLECTION = self.get_game_collection(PATH_TO_RESPONSES)
        print('----------------------------------------------------------------------------------\n')



        # time.sleep(1)
        # print('> preparing team/player profiles ...')
        # # профайлинг игроков и команд в играх
        # self.df_player_profile, self.df_team_profile = self.get_profiles(self.L_COLLECTION)        
        # gc.collect()
        # print('----------------------------------------------------------------------------------\n')


        # # коллекционирование признаков для игр
        # time.sleep(1)
        # print('> collecting features: 1. game info ...')
        # self.add_features__gameinfo(PATH_TO_FEATURES_GAMEINFO)
        # time.sleep(1)
        # print('> collecting features: 2. team history aggregation ...')
        # self.add_features__team(PATH_TO_FEATURES_TEAM)
        # time.sleep(1)
        # print('> collecting features: 3. player history aggregation ...')
        # self.add_features__player(PATH_TO_FEATURES_PLAYER)
        # print('----------------------------------------------------------------------------------\n')

        # del self.df_player_profile, self.df_team_profile
        # gc.collect()

        # time.sleep(1)
        # print('> building features ...')
        # # признаки
        # df_features = self.build_features(PATH_TO_FEATURES_GAMEINFO, PATH_TO_FEATURES_TEAM, PATH_TO_FEATURES_PLAYER)
        # # df_features.to_pickle('df_features.pickle')
        # # df_features_c = pd.read_pickle('df_features.pickle')
        # # df_features = df_features_c.iloc[-5000:]
        # # del df_features_c
        # # gc.collect()

        # time.sleep(1)
        # print('> building targets ...')
        # # целевые переменные
        # L_GAME_IDXS = np.unique(df_features['id'])
        # df_targets = self.build_targets(L_GAME_IDXS) 
        # # df_targets.to_pickle('df_targets.pickle')
        # # df_targets = pd.read_pickle('df_targets.pickle')
        # print('----------------------------------------------------------------------------------\n')

        # time.sleep(1)
        # print('> preparing dataset for ml ...')
        # # подготовка датасета к обучению
        # X, Y = self.prepare_data(df_targets, df_features)
        # del df_targets, df_features
        # gc.collect()
        # # X.to_pickle('X.pickle'), Y.to_pickle('Y.pickle')
        # print('----------------------------------------------------------------------------------\n')   

        
        # time.sleep(1)
        # print('> updating dataset version  ...')
        # # обновление датасета для обучения
        # self.update_dataset(X, Y, PATH_TO_DATASET)
        # print('----------------------------------------------------------------------------------\n')   

        time.sleep(1)
        print('> running ml pipelines  ...')
        # результаты для целевых переменных
        self.L_RUN_RESULTS = []
        # целевые переменные
        l_keys = [
            'TARGET_start_ct__win',
            'TARGET_total__m__27', 'TARGET_total__m__28',
            'TARGET_total__m__29', 'TARGET_total__m__30'
        ]
        # для каждой целевой переменной
        for i, key in enumerate(l_keys):
            k = '_'.join(key.split('_')[1:])
            print('> iter#{}/{}. {}'.format(i+1, len(l_keys), k))
            # выполняем пайплайн
            d_run_result = self.run_ml_pipeline(PATH_TO_DATASET, key)            
            # коллекционируем результат
            self.L_RUN_RESULTS.append(d_run_result)            
            del d_run_result
            gc.collect()
        with open('L_RUN_RESULTS.pickle', 'wb') as f:
            pickle.dump(self.L_RUN_RESULTS, f)
        # with open('L_RUN_RESULTS.pickle', 'rb') as f:
        #     self.L_RUN_RESULTS = pickle.load(f)
        print('----------------------------------------------------------------------------------\n')   

        self.PATH_TO_DATASET = PATH_TO_DATASET

        return self

    def transform(self, PATH_OUT):

        maps_str= """
        Vertigo
        Inferno
        Nuke
        Dust2
        Mirage 
        Ancient 
        Overpass
        """

        teams_str=\
        """
        Natus Vincere
        Gambit
        NIP
        Vitality
        G2
        FaZe
        Heroic
        Astralis
        Virtus.pro
        OG
        ENCE
        BIG
        Liquid
        Movistar Riders
        Copenhagen Flames
        FURIA
        mousesports
        forZe
        Spirit
        Entropiq
        Complexity
        Sinners
        Fiend
        SKADE
        GODSENT
        fnatic
        Lyngby Vikings
        DBL PONEY
        paiN
        Dignitas
        Bad News Bears
        Evil Geniuses
        TeamOne
        Sharks
        00Nation
        Bravos
        Havan Liberty
        MIBR
        FATE
        eSuba
        ECLOT
        Entropiq Prague
        OPAA
        AaB
        MASONIC
        Tricked
        AGF
        Astralis Talent
        HAVU
        KOVA
        hREDS
        SJ
        LDLC
        Sprout
        cowana
        NLG
        TTC
        BIG Academy
        AGO
        Wisla Krakow
        Anonymo
        Izako Boars
        HONORIS
        PACT
        sAw
        SAW Youngsters
        FTW
        OFFSET
        Nexus
        ONYX
        4glory
        Enterprise
        GamerLegion
        Galaxy Racer
        Apeks
        AURA
        Young Ninjas
        Lilmix
        Eternal Fire
        Sangal
        Endpoint
        1WIN
        K23
        INDE IRAE
        AVE
        Singularity
        NAVI Junior
        Spirit Academy
        VP.Pridigy
        Trasko
        EC Kyiv
        B8
        TyLoo
        ViCi
        Lynn Vision
        Invictus
        Checkmate
        D13
        Renegades
        """
        #############################################################################################################

        time.sleep(1)
        print('> preparing team/map to use ...')
        d_map_id2name = {}
        d_team_id2name = {}
        for d_rsp in tqdm.tqdm(self.L_COLLECTION):
            try:
                d_map_id2name[d_rsp['map']['id']] = str.lower(d_rsp['map']['name']).strip()
                for t in d_rsp['teams']:
                    d_team_id2name[t['id']] = str.lower(t['name']).strip()
            except:
                pass
        d_map_name2id={v:k for k, v in d_map_id2name.items()}
        d_team_name2id={v:k for k, v in d_team_id2name.items()}
        L_maps2use = [str.lower(x.strip()) for x in maps_str.split('\n')][1:-1]
        L_map_id2use = [d_map_name2id[map_name] for map_name in L_maps2use]
        L_teams2use = [str.lower(x.strip()) for x in teams_str.split('\n')][1:-1]
        L_team_id2use = []
        for team_name in L_teams2use:
            if team_name in d_team_name2id.keys():
                L_team_id2use.append(d_team_name2id[team_name])

        dataset_c = pd.read_pickle(self.PATH_TO_DATASET)
        dataset = dataset_c.iloc[-5000:]
        del dataset_c
        gc.collect()
        data4model = dataset[['FEATURE_map_id', 'FEATURE_start_ct__team_id', 'FEATURE_start_t__team_id']]
        print('----------------------------------------------------------------------------------\n')   

        

        time.sleep(1)
        print('> preparing forecasts ...')
        L_answer = []
        for d in tqdm.tqdm(self.L_RUN_RESULTS):
            
            # признаки
            L_all_features = d['features'].tolist()
            L_fs4ct, L_fs4t, L_fs4glb = [], [], []
            for f in L_all_features:
                if 'start_ct' in f:
                    L_fs4ct.append(f)
                elif 'start_t' in f:
                    L_fs4t.append(f)
                else:
                    L_fs4glb.append(f)
            assert (len(L_fs4ct)+len(L_fs4t)) == len(L_all_features)
            dataset_c = dataset[L_all_features]

            # данные
            X_train, X_test, y_train, y_test = train_test_split(
                                            dataset[L_all_features], dataset['TARGET_start_ct__win'], 
                                            test_size=.1, shuffle = False
                                            )   
            print('\t> preparing model ensemble ...')
            # ансамбль моделей
            n_models = 10
            L_proba = []   
            L_model = []      
            for _ in tqdm.tqdm(range(n_models)):
                params = d['params'].copy()
                params['random_state'] = _
                model = cb.CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set = (X_test, y_test), early_stopping_rounds=100)
                L_proba.append(model.predict_proba(X_test)[:, 1])
                L_model.append(model)
            L_proba = np.array(L_proba)            
            L_model = np.array(L_model)
            L_idxs = np.arange(n_models)
            best_score = -np.inf
            for n in range(1, n_models+1):
                for cmb in itertools.combinations(L_idxs, n):
                    cmb_idxs = list(cmb)
                    proba_avg = np.c_[L_proba[cmb_idxs]].T.mean(1)                   
                    assert len(proba_avg)==len(X_test)
                    score_blend = roc_auc_score(y_test, proba_avg)
                    if score_blend>best_score:
                        best_score = score_blend
                        L_blend_model = L_model[cmb_idxs]

            time.sleep(1)
            print('\t> calucating forecasts ...')
            # прогноз
            for map_id, subdf in tqdm.tqdm(data4model.groupby('FEATURE_map_id')):

                if map_id in L_map_id2use:

                    L_start_ct_id = np.intersect1d(L_team_id2use, subdf['FEATURE_start_ct__team_id'])
                    L_start_t_id = np.intersect1d(L_team_id2use, subdf['FEATURE_start_t__team_id'])
                    
                    for start_ct_id in tqdm.tqdm(L_start_ct_id):
                        for start_t_id in L_start_t_id:
                            if start_ct_id!=start_t_id:
                                
                                df_ct = subdf.query('FEATURE_start_ct__team_id==@start_ct_id')
                                df_t = subdf.query('FEATURE_start_t__team_id==@start_t_id')
                                
                                if (len(df_ct)!=0)&(len(df_t)!=0):
                                    
                                    game_id4ct = df_ct.index[-1]
                                    game_id4t = df_t.index[-1]

                                    d_fs4gm = {}
                                    d_fs4gm.update(dataset_c.loc[game_id4ct, L_fs4ct].to_dict())
                                    d_fs4gm.update(dataset_c.loc[game_id4t, L_fs4t].to_dict())
                                    x_new = pd.DataFrame.from_records([d_fs4gm])[L_all_features]                                    
                                    proba = np.mean([model.predict_proba(x_new)[0][1] for model in L_blend_model])

                                    d_answer = {}
                                    d_answer['target'] = d['target']
                                    d_answer['map'] = d_map_id2name[map_id]
                                    d_answer['start_ct'] = d_team_id2name[start_ct_id]
                                    d_answer['start_t'] = d_team_id2name[start_t_id]
                                    d_answer['probability'] = proba

                                    L_answer.append(d_answer)
            del dataset_c
            gc.collect()

        df_answer = pd.DataFrame.from_records(L_answer)
        del L_answer
        gc.collect()
        df_answer.to_csv(PATH_OUT)
        del df_answer
        gc.collect()

        return self

# %%
# директория с коллекцией респонсов
PATH_TO_RESPONSES = 'L_games_collection'
# директория с коллекцией признаков для игр (информация об игре)
PATH_TO_FEATURES_GAMEINFO = r'D:\\features_gameinfo'
# директория с коллекцией признаков для игр (командная статистика)
PATH_TO_FEATURES_TEAM = r'D:\\features_team'
# директория с коллекцией признаков для игр (статистика игроков)
PATH_TO_FEATURES_PLAYER = r'D:\\features_player'
# датасет
PATH_TO_DATASET = 'dataset.pickle'
# таблица с прогнозами
PATH_TO_FORECAST = 'forecast.txt'

if __name__ =='__main__':
    # модель
    csgo_ml = CsgoOutcomePredictor()
    # обучение
    csgo_ml.fit(
        PATH_TO_RESPONSES,
        PATH_TO_FEATURES_GAMEINFO,
        PATH_TO_FEATURES_TEAM,
        PATH_TO_FEATURES_PLAYER,
        PATH_TO_DATASET
    )
    # предсказания
    csgo_ml.transform(PATH_TO_FORECAST)
