# -*- coding: utf-8 -*-

import pandas as pd
import pickle, time
import networkx as nx

from features.string_features import *
from features.simple_features import *
from features.hrdag_features import *
from features.arabic_rules import *
from utils.parser_utils import nameCleanerFunction
from utils.clustering import fcluster_one_cc


# Importing Data
def import_dataset(filename, encoding='utf-8', separator='|'):
	return pd.DataFrame.from_csv(filename, encoding=encoding, sep=separator)


# Checking Data
def df_column_cleaner(df, id_cols_input, name_cols_input, dod_cols_input, loc_cols_input):

	list_check = [(id_cols_input, ['hash_1', 'hash_2']),
					(name_cols_input, ['name_1', 'name_2']),
					(dod_cols_input, ['date_of_death_1', 'date_of_death_2']),
					(loc_cols_input, ['location_1', 'location_2'])]

	output_col_name = []

	for input_list, output_list in list_check:
		if input_list != output_list:
			for j, input_name in enumerate(input_list):
				df[output_list[j]] = df[input_name]
		output_col_name += output_list

	return df[output_col_name]


def data_cleaner(df, arabic_cols=['name_1', 'name_2', 'location_1', 'location_2'], na_removal=True):

	df_copy = df.copy()

	for col in arabic_cols:
		df_copy[col] = df[col].apply(lambda x: nameCleanerFunction(x))
	if na_removal:
		df_copy.dropna(axis=0, how='any')
	return df_copy


# Create Arabic Rule Versions of names

def apply_arabic_rules(df, arabic_rules_dict, col_names=['name_1', 'name_2']):

	for name_col in col_names:
		for suff, arabic_func in arabic_rules_dict.iteritems():
			df[name_col+suff] = df[name_col].apply(arabic_func)

	return df


# Apply features to data
def apply_features(df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict):

	#hrdag features applied
	for func_name, tuple_obj in hrdag_feat_dict.iteritems():
		func, m1, m2 = tuple_obj
		df[func_name] = df[[m1,m2]].apply(lambda x: func(x[0], x[1]), 1)

	#both simple and strings features applied
	suff_vec = [''] + arabic_rules_dict.keys()
	full_features_dict = dict(simple_feat_dict.items() + string_feat_dict.items())

	for func_name, tuple_obj in full_features_dict.iteritems():
		func, m1, m2 = tuple_obj
		for suff in suff_vec:
			df[func_name + suff] = df[[m1+suff, m2+suff]].apply(lambda x: func(x[0], x[1]), 1)

	return df


# Drop all not-necessary data
def select_features_for_classification(df):

	feature_list = pickle.load(open('data/col_names_class.pkl', 'rb'))
	df_selected = df[feature_list]
	
	if isinstance(df_selected, pd.DataFrame):
		df_selected = df_selected.as_matrix()

	return df_selected


# Run Classification and attach results (match or not match)
# XGboost object need to be imported here

def run_classification(X_matrix, xgboost_filename = 'data/xgboost_class_model.pkl'):

	xgboost_model = pickle.load(open(xgboost_filename, 'rb'))
	return xgboost_model.predict_proba(X_matrix)[:,1]


# HDF file format
# Create HDF format

def create_hdf_file(df, hash_cols = ['hash_1', 'hash_2'], thresh_col='xgb_prob'):
	
	hash_cols.append(thresh_col)
	save_df = df[hash_cols]

	hdf_file = pd.HDFStore('data/classified-pairs.h5', mode='w',)

	#https://gist.github.com/hunterchung/6478cb25e9d774581de9
	types = save_df.apply(lambda x: pd.api.types.infer_dtype(x.values))
	save_df_copy = save_df.copy()
	for col in types[types=='unicode'].index:
	  save_df_copy[col] = save_df[col].astype(str)
	save_df_copy.columns = [str(c) for c in save_df_copy.columns]

	hdf_file.put('pairs', save_df_copy, format='table', data_columns=True, econding='utf-8')
	hdf_file.close()


# Patrick's code
def clustering_step(hashids_set, hdf_filename = 'data/classified-pairs.h5', threshold=0.5, mock=True):

	if mock:

		hashids_set = set(['A', 'B', 'C', 'D', 'E', 'F'])
		cp = pd.DataFrame.from_records([
		        ('A', 'B', 0.9),
		        ('A', 'C', 0.4),
		        ('A', 'D', 0.6),
		        ('A', 'E', 0.3),
		        ('B', 'C', 0.6),
		        ('C', 'F', 0.1 ),
		        ('E', 'F', 0.97),
		        ('D', 'E', 0.95),
		        ('D', 'F', 0.65)], 
		        columns=['hash_1', 'hash_2', 'xgb_prob'])

		cp.set_index(['hash_1', 'hash_2'], drop=False, inplace=True)
		#clusters_t = fcluster_one_cc(list(hashids_set), cp, 'xgb_prob', verbose=False)
		#print(clusters_t)

	else:
		cp = pd.read_hdf(hdf_filename, 'pairs')
		cp.set_index(['hash_1', 'hash_2'], drop=False, inplace=True)
		print("there are {} classified pairs".format(len(cp)))
		print('ready.')

	G = nx.Graph()
	G.add_nodes_from(hashids_set)  # make sure every record is in a component

	positives = cp.xgb_prob > threshold
	hashpairs = zip(cp.loc[positives].hash_1, cp.loc[positives].hash_2)
	positive_pairs = [(str(h1), str(h2)) for h1, h2 in hashpairs]

	print("number of positive pairs={}".format(len(positive_pairs)))
	G.add_edges_from(positive_pairs)
	connected_components = [c for c in nx.connected_components(G)]
	print("number of connected_components={}".format(len(connected_components)))
	#del G, positives, positive_pairs, hashpairs
	print connected_components

	clusters = list()
	for cc in connected_components:
		print cc, 'CONNECTED COMPONENT'
		start = time.time()
		i_clusters = fcluster_one_cc(cc, cp, 'xgb_prob', verbose=True)
		clusters.extend(i_clusters)
		elapsed = time.time() - start
		print("w len(cc)={}, time = {:3.1f}s".format(len(cc), elapsed))
	
	clusters = [x[0] if isinstance(x, list) and len(x)==1 else x for x in clusters]
	
	return clusters
	


if __name__ == '__main__':

	id_name_col = ['id_1', 'id_2']
	input_name_col = ['match_1','match_2']
	input_dod_col = ['date_of_death_1', 'date_of_death_2']
	input_loc_col = ['location_1', 'location_2']
	
	mock_df_filename = 'data/mock_dataset_hrdag_pipeline_2.csv'
	df = import_dataset(mock_df_filename)
	cleaned_df = df_column_cleaner(df, id_name_col, input_name_col, input_dod_col, input_loc_col)
	cleaned_df = data_cleaner(cleaned_df)

	hash_1_list = cleaned_df['hash_1'].astype(str).values.tolist()
	hash_2_list = cleaned_df['hash_2'].astype(str).values.tolist()
	hashids_set = set(hash_1_list+hash_2_list)

	full_df = apply_arabic_rules(cleaned_df, arabic_rules_dict=arabic_rules_dict)
	features_df = apply_features(full_df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict)

	full_df['xgb_prob'] = run_classification(select_features_for_classification(features_df))
	create_hdf_file(full_df)

	clustering_step(hashids_set=hashids_set)






