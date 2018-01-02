# -*- coding: utf-8 -*-
def Astr(string):
    """
    This function encodes in utf-8 a ASCII string with Arabic character in order for it to be
    compatible with the string formatting in Python.

    INPUT
    string - ASCII string with arabic characters

    OUTPUT
    The input string encoded with utf-8
    """

    return unicode(string, encoding='utf-8')


def readjustSpacesInString(string):
    """
    This function removes the double spaces in a string, on top of the spaces at the beginning/end.

    INPUT
    string - any kind of string, works with Arabic string if utf-8 encoded

    OUTPUT
    The same string without double spaces, or single spaces in the beginning/end of string.
    """

    return ' '.join([x for x in string.split(' ') if x])


def cleanConcatFunction(string, concat_symbol='_'):
    """
    This function removes the concatenation done by the concatenateFunction below on a string.
    Useful just for output purposes

    INPUT
    string - a string which could have been some words being concatenated during the parsing process

    OUTPUT
    The same string without the concatenation - i.e. without "_" of whatever the concatenation symbol was
    """

    return string.replace(concat_symbol, ' ')


def recognizedNameFunction(arab_str, arab_list):
    '''
    This is an ad-hoc function built in order to check whether an arabic string is in a list of other, maybe particular, arabic strings.

    INPUT:
    - arab_str: the arabic string

    OUTPUT:
    - T/F - boolean output whether the string is in the provided list or not
    '''

    return arab_str in [Astr(x) for x in arab_list]


def isolateNextWordDictKeys(arab_string, dict_keys):
    """
    This function is used to find whether there are in the string words that signal a particular name structure
    and returning both the structure and the adjusted string. If the particular name structure is not found
    then it is returned as a null, with the arab_string left untouched.

    INPUT
    arab_string - The arabic string taken into consideration
    dict_keys - The flag for the name structure were set as a key of a dictionary, using the respective values
          as the translation of the flag.

    OUTPUT
     All the outputs are adjusted for double/unnecessary spacing.
    extraction - This is the part of the string referring to the particular name structure
    before - This is the part of the string that was before the particular name structure
    arab_string - This is the originary string pruned of "extraction", so without the particular name
          structure.
    """

    extraction, before = '', ''
    
    for prefix in dict_keys:
        if prefix in arab_string:

            arab_string_prefcorr = arab_string.replace(prefix, prefix[:-1]+'~').split(' ')
            nameparts_pref = [1 if '~' in x else 0 for x in arab_string_prefcorr]

            extraction = '|'.join([cleanConcatFunction(x,concat_symbol='~') for i,x in enumerate(arab_string_prefcorr) 
                            if nameparts_pref[i]==1])

            if nameparts_pref.index(1) >0:
                before = ' '.join([arab_string_prefcorr[i] for i in range(nameparts_pref.index(1))])

            arab_string = [x for i,x in enumerate(arab_string_prefcorr) if nameparts_pref[i]==0]
            arab_string = '' if not arab_string else ' '.join(arab_string)

    return extraction, before, arab_string
            

def concatenateFunction(arab_string, elem_list, isolated_flag = True):
    """
    This function is used to identify whether there are words in the arabic string which are actually to
    be considered together with the word afterwards as they do not mean anything by themselves.
    As example of that is the word "Abdul", which means "Servant of", and so needs to be considered
    together with the name afterwards.

    INPUT
    arab_string - The arabic string
    elem_list - A list of utf-8 encoded arabic strings with such words. 
       NB: It is necessary to have the such words as word+single space, so if we think the word 
       "AL" should be concatenated with the one afterward we should pass "AL " rather than "AL".
       This is to guarantee the words starting with the string would not be picked up.
    isolated_flag - This guarantees that the word we are looking for is not the end or part of a bigger
       word.

    OUTPUT
    mod_arab_string - The arabic string, with the words concatenated with a "_" if any of such words
       were found.
    """

    mod_arab_string = arab_string
    
    def returnIsolated(elem, arab_string):
        output = True
        if elem in arab_string:
            index = arab_string.find(elem)
            if index >0:
                output = arab_string[index-1] == ' '
        return output
    
    for elem in elem_list:
        if elem in mod_arab_string and (not isolated_flag or returnIsolated(elem, arab_string)):

            mod_arab_string = mod_arab_string.replace(elem, elem[:-1]+'_')
    
    return mod_arab_string
