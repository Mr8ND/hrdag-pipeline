# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from parser_utils import Astr, readjustSpacesInString, isolateNextWordDictKeys, concatenateFunction, cleanConcatFunction, recognizedNameFunction
from parser_objs import name_part_dict, concat_list, laqab_F, laqab_M, nisbah_M
#from names_folder.specific_names import createVectorNamesFromFile


def arabicNameParser(arab_str, name_dict = name_part_dict, conc_list = concat_list, print_flag= True): #, filepath_names_folder = '../'):

    """
    This function is the main arabic name parser function.

    INPUT:
    arab_str - Single string in Arabic, encoded in utf-8
    name_dict - Dictionary with the signaling words for each name part available
    concat_list - List with all the words which should be attached to the word after them, which are
           generally not indicative of a specific name part.

    OUTPUT:
    List of Tuple containing the following pairs, where "xxx" is the relative name or '' if not found:
     (ISM, "xxx")
     (LAQAB, "xxx")
     (NISBAH, "xxx")
     (LAQAB/NISBAH, "xxx")
     (NASAB, "xxx")
     (KUNYA, "xxx")
     (OTHER, "xxx")
    """

    #We initialize the different variables here and the particular name files we have stored.
    
    ism, laqab, nasab, kunya, nisbah, laqnisb, other = '', '', '', '', '', '', ''
    #laqab_M = createVectorNamesFromFile(filepath_names_folder + 'names_folder/laqab_M.txt')
    #laqab_F = createVectorNamesFromFile(filepath_names_folder + 'names_folder/laqab_F.txt')
    #nisbah_M = createVectorNamesFromFile(filepath_names_folder + 'names_folder/nisbah_M.txt')
    
    #A little bit of pre-processing.
    #We first of all want to identifyif the string is made of all arabic characters or not.
    #We remove the space (position 32) from the plausible arabic characters.

    unicode_order_str = [1 if (ord(c)<128 and ord(c)!=32) else 0 for c in arab_str]
    if sum(unicode_order_str)>0 and print_flag:
        print 'Your string contains not only Arabic characters. The following were identified: %s'%(
            ' | '.join(['*'+str(x)+'*' for j,x in enumerate(arab_str) if unicode_order_str[j]==1]))
    if isinstance(arab_str, str):
        arab_str = Astr(arab_str)

    #The are some words in Arabic, like Abdul, that means nothing alone, but they attach to the successive word.
    #They should be a single word, the function below does that in the arabic string taken into consideration.
    
    arab_str = concatenateFunction(arab_str, conc_list)
    
    
    #Now the first step for the name is to identify either the Nasab or Kunya.
    #If we identify any of them and there are words before, the first one is an ism.
    #This does not work for a Nisbah though, we will just include it the nisbah if we can
    #identify it.

    def identifyNamePartFunction(arab_string, namepart_list):
        """
        This function is a wrapper around the isolateNextWordDictKeys function, with the extra that
        cleans up the name structure picked up by that function, if any.

        INPUT
        arab_string - The arabic string in input
        namepart_list - The string to be passed into the isolateNextWordDictKeys as they signal a particular
             name structure.

        OUTPUT
        np_str - The string with the particular name structure identified from the string.
        bef_np - The part before such particular structure, if any was identified.
        arab_string - The string pruned of the particular name structure, if any was identified.
        """

        np_str, bef_np, arab_string = isolateNextWordDictKeys(arab_string, namepart_list)
        
        if np_str:
            np_str = readjustSpacesInString(np_str.replace('|',' '))
        
        return np_str, bef_np, arab_string


    def identifyIsmFromNPFunction(before_string, arab_string, ism_part):
        """
        This function identify the ism if the ism was not found before, pruning the string from it.

        INPUT
        before_string - This is the part of the string before a particular name structure. Can be empty
        arab_string - This is the string, either full or pruned of a particular name structure.
             Can be empty, but if before_string is not empty then arab_string contains at least before_string.
        ism_part - The ism of the name. Can be empty


        OUPUT
        ism_part - The ism of the name, if it was given empty and the before_string contained some words.
        arab_string - The arab string pruned of the ism, if there was any.
        """

        if before_string and not ism_part:
            ism_part = before_string.split(' ')[0]
            arab_string = arab_string[len(ism_part)+1:]

        return ism_part, arab_string


    nasab, bef_nasab, arab_str = identifyNamePartFunction(arab_str, name_part_dict['nasab'].keys())
    ism, arab_str = identifyIsmFromNPFunction(bef_nasab, arab_str, ism)

    kunya, bef_kunya, arab_str = identifyNamePartFunction(arab_str, name_part_dict['kunya'].keys())
    ism, arab_str = identifyIsmFromNPFunction(bef_kunya, arab_str, ism)

    nisbah, bef_nisbah, arab_str = identifyNamePartFunction(arab_str, name_part_dict['nisbah'].keys())
    

    #If ISM has not been found now the fist word is taken to be the ism.
    #This is the time also to identify whether the ism does exist in the name.
    
    if not ism:
        if len(arab_str)==0 and print_flag:
            print 'The string does not seem to contain any ISM.'
        else:
            ism = arab_str.split(' ')[0]
            arab_str = arab_str[len(ism)+1:]
    
    #Now we are left with identifying general nisbah or laqab.
    #We know what was before both the kunya and nasab, and that should be the laqab.
    #What was after is usually the nisbah.
    
    bef_vec = [bef_nasab, bef_kunya]
    flag_names = [1 if bef_part else 0 for bef_part in bef_vec]

    if sum(flag_names)>0 and arab_str:
        bef_part_selected = bef_vec[1] if sum(flag_names)==2 else bef_vec[flag_names.index(1)]

        rem_bef = bef_part_selected.replace(ism+' ', '') if ism != bef_part_selected else ''
        after_str = arab_str.replace(rem_bef+' ', '') if rem_bef != arab_str else ''

        if rem_bef and len(rem_bef)>=1:
            laqab += readjustSpacesInString(rem_bef)
        if after_str and len(after_str)>=1:
            nisbah += after_str.split(' ')[0]
            other += ' '.join(after_str.split(' ')[1:]) if len(after_str.split(' '))>1 else ''
    
    elif arab_str:
        
    #If no nasab or kunya, the two after the name should be either laqab/nisbah.
    #If there are more than two after the ism, then the rest goes in other.
        
        arab_str_noism = arab_str.replace(ism+' ', '')
        n = len(arab_str_noism.split(' '))

        if arab_str_noism and n<=2:
            for word_noism in arab_str_noism.split(' '):

                recon_laq = recognizedNameFunction(word_noism, laqab_M+laqab_F)
                recon_nis = recognizedNameFunction(word_noism, nisbah_M)

                if recon_laq:
                    laqab += word_noism if not laqab else ' '+word_noism
                elif recon_nis:
                    nisbah += word_noism if not nisbah else ' '+word_noism
                else:
                    laqnisb += word_noism if not laqnisb else ' '+word_noism

        elif arab_str_noism:
            for word_noism in arab_str_noism.split(' ')[:2]:

                recon_laq = recognizedNameFunction(word_noism, laqab_M+laqab_F)
                recon_nis = recognizedNameFunction(word_noism, nisbah_M)

                if recon_laq:
                    laqab += word_noism if not laqab else ' '+word_noism
                elif recon_nis:
                    nisbah += word_noism if not nisbah else ' '+word_noism
                else:
                    laqnisb += word_noism if not laqnisb else ' '+word_noism

            other += ' '.join(arab_str_noism.split(' ')[2:])
        
    return [tuple(['ISM', cleanConcatFunction(ism)]),
            tuple(['LAQAB', cleanConcatFunction(laqab)]),
            tuple(['NASAB', cleanConcatFunction(nasab)]), 
            tuple(['KUNYA', cleanConcatFunction(kunya)]),
            tuple(['NISBAH', cleanConcatFunction(nisbah)]),
            tuple(['LAQAB/NISBAH', cleanConcatFunction(laqnisb)]),
            tuple(['OTHER', cleanConcatFunction(other)])]


if __name__ == "__main__":

    #start = timer()
    #for x in range(1):
    print Astr('محمد الفاييت العدواني')
    result = arabicNameParser('محمد الفاييت العدواني')
    #print result
    #end = timer()
    #print '%s seconds taken to perform a single name' %(end - start)
    #print '\n'
    #print result
    for x,y in result:
    	print x,y

