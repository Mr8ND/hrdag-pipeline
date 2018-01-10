# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle, time, argparse, os
import networkx as nx

from features.string_features import string_feat_dict
from features.simple_features import simple_feat_dict
from features.hrdag_features import hrdag_feat_dict
from features.arabic_rules import arabic_rules_dict
from utils.parser_utils import nameCleanerFunction
from utils.clustering import fcluster_one_cc


def save_obj_pickle(obj, directory, filename_out):
	'''
	This function pickles an object in a directory with a given filename.

	INPUTS:
	- obj: the object to be pickled with python
	- directory [str]
	- filename_out [str]
	'''

	output = open(directory + filename_out, 'wb')
	pickle.dump(obj, output)
	output.close()


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

	output_col_name = []
	for input_list, output_list in list_check:

		# Check input columns are in df as expected
		for input_col in input_list:
			if input_col not in df.columns.values:
				raise ValueError('Column %s is not in the df columns.'%(input_col))

		# Change name if necessary
		if input_list != output_list:
			for j, input_name in enumerate(input_list):
				df[output_list[j]] = df[input_name]
		output_col_name += output_list

	df.columns = [str(col) for col in df.columns]
	return df[output_col_name]


def arabic_col_cleaner(df, arabic_cols=['name_1', 'name_2', 'location_1', 'location_2'], na_removal=True):
	'''
	Readjust the spaces and removes the Arabic presentation form (old form of arabic) in arabic written columns.
	Also removes NA values if na_removal is True. Raising a ValueError if columns are not in the dataframe.

	INPUT:
	- df [pd.DataFrame]
	- arabic_cols [str]: list of columns names to be cleaned. Expected to have utf-8 encoded arabic strings
	- na_removal [bool]
	'''

	# Performing a copy of the dataframe to avoid referencing errors within pandas
	df_copy = df.copy()

	for col in arabic_cols:
		if col not in df.columns:
			raise ValueError('Column %s not in the dataframe.'%(col))
		df_copy[col] = df[col].apply(lambda x: nameCleanerFunction(x))

	if na_removal:
		df_copy.dropna(axis=0, how='any')

	return df_copy


def apply_arabic_rules(df, arabic_rules_dict, col_names=['name_1', 'name_2']):
	'''
	Creates the name columns which have been modified according to the different arabic rules.
	Raises a ValueError if the target column names are not in the df.

	INPUT:
	- df [pd.DataFrame]
	- arabic_rules_dict [dict]: dictionary in which the value is a function which applies the arabic rule to a string. The key will be added to the column name as suffix (at the bottom)
	- col_names [list]: list of columns to which the arabic rules will be applied

	OUTPUT:
	df with the name columns to which the arabic rules have been applied.
	'''

	for col in col_names:
		if col not in df.columns:
			raise ValueError('Column %s not in the dataframe.'%(col))

	for name_col in col_names:
		for suff, arabic_func in arabic_rules_dict.iteritems():
			df[name_col+suff] = df[name_col].apply(arabic_func)

	return df


def apply_features(df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict):
	'''
	Applies the three sets of features - hrdag, simple and string features - to the dataframe through three dictionaries.
	The dictionaries are built such that the key will be the name of the column created in the dataframe, while the value is a tuple in which the first element is the function and the second and third the two names of the columns to which the function will be applied.
	For simple and string features, functions are also applied to the name columns modified according to certain arabic rules. The keys of the arabic rules dictionary are used to identify such columns - and the keys will be included in the name of the features columns created at the bottom.

	INPUTS:
	- df [pd.DataFrame]
	- simple_feat_dict [dict]: dictionary of simple features, key is the function name, value is (function, column1, column2)
	- hrdag_feat_dict [dict]: dictionary of hrdag features, key is the function name, value is (function, column1, column2)
	- string_feat_dict [dict]: dictionary of string features, key is the function name, value is (function, column1, column2)
	- arabic_rules_dict [dict]: dictionary of arabic rules, key is the arabic rule suffix which will be used to identify the modified arabic column name and will be appended to the function applied to such modified name

	OUTPUT:
	same df in input with all hrdag features applied to name columns and all simple and string features applied not only to the name columns, but also to the name columns modified according to the arabic rules.
	'''

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


def select_features_for_classification(df, feature_list_filename='inputs/col_names_class.pkl'):
	'''
	This function imports the column names which will be used for classification and selects them in the df.
	Raises an IOError if the file with the column names is not found.
	Raise a ValueError if one of the column names is not in the df.
	Casts the df as a numpy matrix.

	INPUT:
	- df [pd.DataFrame]
	- feature_list_file [str]: pickle file in which the column names which will be used as features are saved

	OUTPUT:
	df only with the selected columns, casted as a matrix.
	'''

	if not os.path.exists(feature_list_filename):
		raise IOError('File %s does not exist.'%(feature_list_filename))

	feature_list = pickle.load(open(feature_list_filename, 'rb'))
	
	for feat in feature_list:
		if feat not in df.columns.values:
			raise ValueError('Feature %s is not in the df'%(feat))
	df_selected = df[feature_list]

	return df_selected.as_matrix()


def run_classification(X_matrix, xgboost_filename='inputs/xgboost_class_model.pkl'):
	'''
	This function imports the xgboost model and obtain the probability of a match from the X_matrix.
	The columns are supposed to have been arranged in the correct order.
	Raises an IOError if the filename with the xgboost is not found.
	Raises a TypeErorr if the X_matrix is not a np.ndarray

	INPUT:
	- X_matrix [np.ndarray]
	- xgboost_filename [str]

	OUTPUT:
	Probability of a match for each of the name pairs.
	'''

	if not os.path.exists(xgboost_filename):
		raise IOError('File %s does not exist.'%(xgboost_filename))

	if not isinstance(X_matrix, np.ndarray):
		raise TypeError('X_matrix is not a np.ndarray, currently %s'%(type(X_matrix)))

	xgboost_model = pickle.load(open(xgboost_filename, 'rb'))
	return xgboost_model.predict_proba(X_matrix)[:,1]


def create_hdf_file(df, hash_col_1='hash_1', hash_col_2='hash_2', thresh_col='xgb_prob', filename_out='inputs/classified-pairs.h5', dataframe_name_hdf='pairs'):
	'''
	This function creates an HDF file from a dataframe, using two columns as hashes and a third one as matching probability.
	Raises a ValueError if the columns are not in the dataframe.

	INPUT:
	- df [pd.DataFrame]
	- hash_col_1 [str]
	- hash_col_2 [str]
	- thresh_col [str]: matching probability column
	'''
	
	cols_sel = [hash_col_1, hash_col_2, thresh_col]
	for col in cols_sel:
		raise ValueError('Column %s is not in the dataframe.'%(col))
	save_df = df[cols_sel]

	# Open HDF file
	hdf_file = pd.HDFStore(filename_out, mode='w')

	# The unicode formatting creates problem with HDF files.
	# The following casts as strings the columns which are unicode (gist.github.com/hunterchung/6478cb25e9d774581de9)
	types = save_df.apply(lambda x: pd.api.types.infer_dtype(x.values))
	save_df_copy = save_df.copy()
	for col in types[types=='unicode'].index:
	  save_df_copy[col] = save_df[col].astype(str)
	save_df_copy.columns = [str(c) for c in save_df_copy.columns]

	# Adding the dataframe with the name dataframe_name_hdf
	hdf_file.put(dataframe_name_hdf, save_df_copy, format='table', data_columns=True, econding='utf-8')
	hdf_file.close()


def classify_step(data_filename, id_col, name_col, dod_col, loc_col):
	'''
	This function is the main wrapper for the classification step.

	INPUT:
	- data_filename [str]
	- id_col [str]: root of the id column in the dataframe. The id columns would then be searched by default by appending '_1' and '_2' at the bottom. E.g. if id_col is "hash", the id columns will be looked for as "hash_1" and "hash_2"
	- name_col [str]: root of the name column in the dataframe. Same rationale as for id_col applies
	- dod_col [str]: root of the name column in the dataframe. Same rationale as for id_col applies
	- loc_col [str]: root of the name column in the dataframe. Same rationale as for id_col applies
	'''

	# Imports the dataframe
	if not os.path.exists(data_filename):
		raise IOError('File %s does not exist.'%(data_filename))
	df = import_dataset(data_filename)

	# Formats the name columns in the right way and cleans them
	id_name_col = [id_col +'_1', id_col +'_2']
	input_name_col = [name_col +'_1', name_col +'_2']
	input_dod_col = [dod_col +'_1', dod_col +'_2']
	input_loc_col = [loc_col +'_1', loc_col +'_2']
	cleaned_df = df_column_cleaner(df, id_name_col, input_name_col, input_dod_col, input_loc_col)
	cleaned_df = arabic_col_cleaner(cleaned_df)

	# Saves the Hash IDs set to a Pickle file
	hash_1_list = cleaned_df['hash_1'].astype(str).values.tolist()
	hash_2_list = cleaned_df['hash_2'].astype(str).values.tolist()
	hashids_set = set(hash_1_list+hash_2_list)
	save_obj_pickle(obj=hashids_set, directory='inputs/', filename_out='hashid_set.pkl')

	# Applies the classification step
	full_df = apply_arabic_rules(cleaned_df, arabic_rules_dict=arabic_rules_dict)
	features_df = apply_features(full_df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict)
	full_df['xgb_prob'] = run_classification(select_features_for_classification(features_df))
	
	#Creates an HDF file
	create_hdf_file(full_df)

	print 'Classification Done'	


# Patrick's code
def clustering_step(hashid_filename, input_pairs_filename, cp=None, threshold=0.5, mock=True, dataframe_name='pairs'):

	if input_pairs_filename:
		cp = pd.read_hdf(input_pairs_filename, dataframe_name)
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