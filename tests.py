import unittest2 as unittest
import pandas as pd
from main import clustering_step, save_obj_pickle, import_dataset, df_column_cleaner, arabic_col_cleaner, apply_arabic_rules, select_features_for_classification, run_classification
from features.arabic_rules import arabic_rules_dict


mock_filename = 'data/mock_dataset_hrdag_pipeline_2.csv'


class MainFunctionTests(unittest.TestCase):

	def test_clustering_step_pbexample_function(self):

		hashids_set = set(['A', 'B', 'C', 'D', 'E', 'F'])
		cp = pd.DataFrame.from_records([('A', 'B', 0.9), ('A', 'C', 0.4),
				('A', 'D', 0.6), ('A', 'E', 0.3), ('B', 'C', 0.6),
				('C', 'F', 0.1 ), ('E', 'F', 0.97),	('D', 'E', 0.95),
				('D', 'F', 0.65)], 
				columns=['hash_1', 'hash_2', 'xgb_prob'])
		cp.set_index(['hash_1', 'hash_2'], drop=False, inplace=True)

		save_obj_pickle(obj=hashids_set, directory='inputs/', filename_out='mock_hashids.pkl')
		self.assertEqual([['E', 'D', 'F'], ['A', 'C', 'B']], clustering_step('inputs/mock_hashids.pkl', None, cp=cp))


	def test_import_dataset_function(self):

		self.assertRaises(IOError, import_dataset, 'clearly_not_a_file')
		self.assertRaises(IOError, import_dataset, 'inputs/clearly_not_a_file')


	def test_df_column_cleaner_function(self):

		df_mock = import_dataset(mock_filename)

		self.assertRaises(ValueError, df_column_cleaner, df_mock, ['hash_1', 'hash_2'], ['match_1', 'match_2'], 
			['date_of_death_1','date_of_death_2'], ['location_1', 'location_2'])
		self.assertRaises(ValueError, df_column_cleaner, df_mock, ['id_1', 'id_2'], ['matching_1', 'matching_2'], 
			['date_of_death_1','date_of_death_2'], ['location_1', 'location_2'])
		self.assertRaises(ValueError, df_column_cleaner, df_mock, ['id_1', 'id_2'], ['match_1', 'match_2'], 
			['date_of_dying_1','date_of_dying_2'], ['location_1', 'location_2'])
		self.assertRaises(ValueError, df_column_cleaner, df_mock, ['id_1', 'id_2'], ['match_1', 'match_2'], 
			['date_of_death_1','date_of_death_2'], ['loc_1', 'loc_2'])

		expected_col_names = ['hash_1', 'hash_2', 'name_1', 'name_2', 'date_of_death_1', 'date_of_death_2', 'location_1', 'location_2']
		self.assertEqual(set(expected_col_names), set(df_column_cleaner(df_mock, ['id_1', 'id_2'], ['match_1', 'match_2'], 
			['date_of_death_1','date_of_death_2'], ['location_1', 'location_2']).columns.values))


	def test_arabic_col_cleaner_function(self):

		df_mock = import_dataset(mock_filename)
		self.assertRaises(ValueError, arabic_col_cleaner, df_mock, ['not_a_col', 'not_a_col_either'])


	def test_apply_arabic_rules_function(self):

		df_mock = import_dataset(mock_filename)
		self.assertRaises(ValueError, apply_arabic_rules, df_mock, arabic_rules_dict, ['not_a_col', 'not_a_col_either'])


	def test_select_features_for_classification_function(self):
		
		df_mock = import_dataset(mock_filename)
		self.assertRaises(IOError, select_features_for_classification, df_mock, 'not_a_file')
		self.assertRaises(IOError, select_features_for_classification, df_mock, 'inputs/not_a_file_either')


	def test_run_classification_function(self):

		df_mock = import_dataset(mock_filename)
		self.assertRaises(IOError, run_classification, df_mock, 'not_a_file')
		self.assertRaises(IOError, run_classification, df_mock, 'inputs/not_a_file_either')

		self.assertRaises(TypeError, run_classification, df_mock)



if __name__ == '__main__':
	unittest.main()