# -*- coding: utf-8 -*-
from datetime import datetime
from sets import Set
from pyjarowinkler import distance as jw_distance
import icu
from datasketch import MinHash


def firstCharFunction(s1,s2,n):
	return int(s1[:n]==s2[:n])

def lastCharFunction(s1,s2,n):
	return int(s1[::-1][:n]==s2[::-1][:n])

def first5CharsFunction(s1,s2):
	return firstCharFunction(s1,s2,5)

def last5CharsFunction(s1,s2):
	return lastCharFunction(s1,s2,5)

def computeJaccardIndexFunction(s1, s2):
	set_1, set_2 = Set(s1.split(' ')), Set(s2.split(' '))
	n = len(set_1.intersection(set_2))
	return n / float(len(set_1) + len(set_2) - n) 

def sameYearAndMonthFunction(dod1, dod2):
	dod1_date = datetime.strptime(dod1, '%Y-%m-%d')
	dod2_date = datetime.strptime(dod2, '%Y-%m-%d')
	return int(dod1_date.year==dod2_date.year and dod1_date.month==dod2_date.month)

def diffDaysFunction(dod1, dod2):
	dod1_date = datetime.strptime(dod1, '%Y-%m-%d')
	dod2_date = datetime.strptime(dod2, '%Y-%m-%d')
	return abs((dod1_date-dod2_date).days)

def jwOnSortedFunction(s1, s2, collator=icu.Collator.createInstance(icu.Locale('de_DE.UTF-8'))):
	s1_s = ''.join(sorted(list(s1), key=collator.getSortKey))
	s2_s = ''.join(sorted(list(s2), key=collator.getSortKey))
	return jw_distance.get_jaro_distance(s1_s, s2_s, winkler=True)

def locSensitiveHashingFunction(s1, s2, tresh=.5):
    set_1, set_2 = Set(s1.split(' ')), Set(s2.split(' '))
    m1, m2 = MinHash(), MinHash()
    for d in set_1:
        m1.update(d.encode('utf-8'))
    for d in set_2:
        m2.update(d.encode('utf-8'))
    return 1 if m1.jaccard(m2)>tresh else 0


hrdag_feat_dict = {'first_5_chars_same': (first5CharsFunction, 'name_1', 'name_2'),
				  'last_5_chars_same': (last5CharsFunction, 'name_1', 'name_2'),
				  'jaccard_on_names': (computeJaccardIndexFunction, 'name_1', 'name_2'),
				  'jaccard_on_location': (computeJaccardIndexFunction, 'location_1', 'location_2'),
				  'jarowinkler_on_sorted': (jwOnSortedFunction, 'name_1', 'name_2'),
				  'same_substring_lochashing': (locSensitiveHashingFunction, 'name_1', 'name_2'),
				  'days_diff_death': (diffDaysFunction, 'date_of_death_1', 'date_of_death_2'),
				  'same_year_month_death': (sameYearAndMonthFunction, 'date_of_death_1', 'date_of_death_2')}