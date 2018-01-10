# -*- coding: utf-8 -*-

import pandas as pd
import pickle, time, argparse, os
import networkx as nx

from features.string_features import *
from features.simple_features import *
from features.hrdag_features import *
from features.arabic_rules import *
from utils.parser_utils import nameCleanerFunction
from utils.clustering import fcluster_one_cc


def import_dataset(filename, encoding='utf-8', separator='|'):
	'''
	Imports a csv as pandas dataframe, just a wrapper around pd.DataFrame.from_csv
	INPUT:
	- filename [str]
	- encoding [str]: the encoding of the file - in this case unicode utf-8 by default
	- separator [str]
	'''
	if not os.path.exists(filename):
		raise IOError('File %s does not exist.'%(filename))

	return pd.DataFrame.from_csv(filename, encoding=encoding, sep=separator)


def df_column_cleaner(df, id_cols_input, name_cols_input, dod_cols_input, loc_cols_input):
	'''
	Changes the name of the columns in the input in the expected column names.
	Raises a ValueError if some of the columns are missing.

	INPUTS:
	- df [pd.DataFrame]
	- id_cols_input [list]: list with two strings indicating the column names for the pair ids
	- name_cols_input [list]: list with two strings indicating the column names for the pair names
	- dod_cols_input [list]: list with two strings indicating the column names for the pair date of death
	- loc_cols_input [list]: list with two strings indicating the column names for the pair locations

	OUTPUT:
	- same df as input with the expected column names.
	'''

	# Pairing the input column names with expected column names
	list_check = [(id_cols_input, ['hash_1', 'hash_2']),
					(name_cols_input, ['name_1', 'name_2']),
					(dod_cols_input, ['date_of_death_1', 'date_of_death_2']),
					(loc_cols_input, ['location_1', 'location_2'])]

	# Create new columns if necessary
	output_col_name = []
	for input_list, output_list in list_check:

		for input_col in input_list:
			if input_col not in df.columns.values:
				raise ValueError('Column %s is not in the df columns.'%(input_col))

		if input_list != output_list:
			for j, input_name in enumerate(input_list):
				df[output_list[j]] = df[input_name]
		output_col_name += output_list

	df.columns = [str(col) for col in df.columns]
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

	feature_list = pickle.load(open('inputs/col_names_class.pkl', 'rb'))
	df_selected = df[feature_list]
	
	if isinstance(df_selected, pd.DataFrame):
		df_selected = df_selected.as_matrix()

	return df_selected


# Run Classification and attach results (match or not match)
# XGboost object need to be imported here

def run_classification(X_matrix, xgboost_filename = 'inputs/xgboost_class_model.pkl'):

	xgboost_model = pickle.load(open(xgboost_filename, 'rb'))
	return xgboost_model.predict_proba(X_matrix)[:,1]


# HDF file format
# Create HDF format

def create_hdf_file(df, hash_cols = ['hash_1', 'hash_2'], thresh_col='xgb_prob'):
	
	hash_cols.append(thresh_col)
	save_df = df[hash_cols]

	hdf_file = pd.HDFStore('inputs/classified-pairs.h5', mode='w',)

	#https://gist.github.com/hunterchung/6478cb25e9d774581de9
	types = save_df.apply(lambda x: pd.api.types.infer_dtype(x.values))
	save_df_copy = save_df.copy()
	for col in types[types=='unicode'].index:
	  save_df_copy[col] = save_df[col].astype(str)
	save_df_copy.columns = [str(c) for c in save_df_copy.columns]

	hdf_file.put('pairs', save_df_copy, format='table', data_columns=True, econding='utf-8')
	hdf_file.close()


# Patrick's code
def clustering_step(hashid_filename, input_pairs_filename, cp=None, threshold=0.5, mock=True):

	if input_pairs_filename:
		cp = pd.read_hdf(input_pairs_filename, 'pairs')
		cp.set_index(['hash_1', 'hash_2'], drop=False, inplace=True)

	hashids_set = pickle.load(open(hashid_filename, 'rb'))
		
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
	
	print clusters
	return clusters


def save_obj_pickle(obj, directory, filename_out):
	'''
	This function pickles an object in a directory with a given filename.
	Originally used to save the LSHindex falconn object but apparently it is not possible to pickle them with Python (https://github.com/FALCONN-LIB/FALCONN/issues/80)

	INPUTS:
	- obj: the object to be pickled with python
	- directory [str]
	- filename_out [str]
	'''

	output = open(directory + filename_out, 'wb')
	pickle.dump(obj, output)
	output.close()


def classify_step(id_col, name_col, dod_col, loc_col, data_filename):

	df = import_dataset(data_filename)

	id_name_col = [id_col +'_1', id_col +'_2']
	input_name_col = [name_col +'_1', name_col +'_2']
	input_dod_col = [dod_col +'_1', dod_col +'_2']
	input_loc_col = [loc_col +'_1', loc_col +'_2']
	cleaned_df = df_column_cleaner(df, id_name_col, input_name_col, input_dod_col, input_loc_col)
	cleaned_df = data_cleaner(cleaned_df)

	hash_1_list = cleaned_df['hash_1'].astype(str).values.tolist()
	hash_2_list = cleaned_df['hash_2'].astype(str).values.tolist()
	hashids_set = set(hash_1_list+hash_2_list)
	save_obj_pickle(obj=hashids_set, directory='inputs/', filename_out='hashid_set.pkl')

	full_df = apply_arabic_rules(cleaned_df, arabic_rules_dict=arabic_rules_dict)
	features_df = apply_features(full_df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict)

	full_df['xgb_prob'] = run_classification(select_features_for_classification(features_df))
	create_hdf_file(full_df)

	print 'Classification Done'	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(help='Possible shazam commands', dest='subparser_name')

	parser_classify = subparsers.add_parser('classify', help='Classification of input pairs')
	parser_classify.add_argument('--id_col', action='store', type=str, default='id')
	parser_classify.add_argument('--name_col', action='store', type=str, default='match')
	parser_classify.add_argument('--dod_col', action='store', type=str, default='date_of_death')
	parser_classify.add_argument('--loc_col', action='store', type=str, default='location')
	parser_classify.add_argument('--data_filename', action='store', type=str, default='data/mock_dataset_hrdag_pipeline_2.csv')

	parser_clustering = subparsers.add_parser('clustering', help='CLustering step of classified input pairs')
	parser_clustering.add_argument('--hashid_filename', type=str, action='store', default='inputs/hashid_set.pkl')
	parser_clustering.add_argument('--input_pairs_filename', type=str, action='store', default='inputs/classified-pairs.h5')

	
	argument_parsed_dict = vars(parser.parse_args())
	subparser_sel = argument_parsed_dict['subparser_name']
	del argument_parsed_dict['subparser_name']

	func_dict = {'classify': classify_step, 
				'clustering': clustering_step}

	func_dict[subparser_sel](**argument_parsed_dict)	







