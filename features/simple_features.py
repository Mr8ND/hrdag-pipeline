from ...utils.arabic_soundex_server import arabic_soundex_names
from collections import Counter

def CompContainedFunction(s1,s2):
    output = 0
    s1_part, s2_part = s1.split(' '), s2.split(' ')
    n1, n2 = len(s1_part), len(s2_part)
    if n1 >= n2:
        output= int(all(x in s1_part for x in s2_part))
    elif n2 > n1:
        output = int(all(x in s2_part for x in s1_part))
    return output

def exactMatchFunction(s1,s2):
    return int(s1==s2)

def soundexMatchFunction(s1,s2):
    return int(arabic_soundex_names(s1)==arabic_soundex_names(s2))

def noSpaceMatchFunction(s1,s2):
    return int(s1.replace(' ','')==s2.replace(' ',''))

def ShuffleMatchFunction(s1,s2):
    return int(Counter(s1) == Counter(s2))


simple_feat_dict = {
        'comp_contained_match': (CompContainedFunction, 'name_1', 'name_2'),
        'exact_match': (exactMatchFunction, 'name_1', 'name_2'),
        'soundex_match': (soundexMatchFunction, 'name_1', 'name_2'),
        'nospace_match': (noSpaceMatchFunction, 'name_1', 'name_2'),
        'shuffle_match': (ShuffleMatchFunction, 'name_1', 'name_2')
    }