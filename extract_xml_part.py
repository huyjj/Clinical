import csv, os
from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing 
drop_node = ['provided_document_section/', 'required_header/', 'reference/', 'link/', 'contact/']


def load_disease2icd():
	disease2icd = dict()
	with open('data/diseases.csv', 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	for row in rows:
		disease = row[0]
		icd = row[1]
		disease2icd[disease] = icd 
	return disease2icd 


def walkData(root_node, prefix, result):
    k = prefix + '/' + root_node.tag
    v = root_node.text
    if (v is not None) and (v.strip('\n').strip('\r').strip() != '') :
        k = k.replace('/clinical_study/', '')
        for i in drop_node:
            if i in k:
                return
            
        if k  not in result.keys():
            result[k] = [v]
        else:
            result[k].append(v)

    children_node = list(root_node)
    if len(children_node) == 0:
        return
    for child in children_node:
        walkData(child, prefix = prefix + '/' + root_node.tag, result = result)


def walkData_attrib(root_node, prefix, result):
	'''
	递归读取xml中所有特征
	'''
	k = prefix + '/' + root_node.tag
	v = root_node.text
	att = root_node.attrib
	k = k.replace('/clinical_study/', '')
	if (v is not None) and (v.strip('\n').strip('\r').strip() != '') :
		if k  not in result.keys():
			result[k] = [v]
		else:
			result[k].append(v)

	if att:
		for attri_name in att.keys():
			k_att = k + '{' + f'{attri_name}' + '}'
			if k_att  not in result.keys():
				result[k_att] = [att[attri_name]]
			else:
				result[k_att].append(att[attri_name])

		# print(k_att,result[k_att])

	children_node = list(root_node)
	if len(children_node) == 0:
		return
	for child in children_node:
		walkData_attrib(child, prefix = prefix + '/' + root_node.tag, result = result)
          

def disease2icdcode(conditions, disease2icd):
    icdcode_lst = []
    for disease in conditions:
        icdcode = disease2icd[disease.lower()] if disease.lower() in disease2icd else None
        icdcode_lst.append(icdcode)

    return icdcode_lst
    

def strip_list(lis):
    '''
    '''
    if len(lis) > 1:
        return lis
    else:
        return lis[0]


# def walkData(root_node, prefix, result):
# 	'''
# 	递归读取xml中所有特征
# 	'''
# 	k = prefix + '/' + root_node.tag
# 	v = root_node.text
# 	att = root_node.attrib
# 	k = k.replace('/clinical_study/', '')
# 	if (v is not None) and (v.strip('\n').strip('\r').strip() != '') :
# 		if k  not in result.keys():
# 			result[k] = [v]
# 		else:
# 			result[k].append(v)

# 	if att:
# 		for attri_name in att.keys():
# 			k_att = k + '{' + f'{attri_name}' + '}'
# 			if k_att  not in result.keys():
# 				result[k_att] = [att[attri_name]]
# 			else:
# 				result[k_att].append(att[attri_name])

# 		# print(k_att,result[k_att])

# 	children_node = list(root_node)
# 	if len(children_node) == 0:
# 		return
# 	for child in children_node:
# 		walkData(child, prefix = prefix + '/' + root_node.tag, result = result)
		

All_data = []
for i in range(10):
    All_data.append(pd.read_csv(f'All_data_part{i}.csv', index_col=0, low_memory=False))
    print(len(All_data[i].columns))

All = pd.concat(All_data)
feature = {'feature_name':[], 'Number':[], 'Percent':[], 'Sample1':[], 'Sample2':[], 'Sample3':[], 'Sample4':[]}
for col in All.columns:
    if All[col].notna().sum() < 4:
        continue
    not_na_value = All[col][All[col].notna()].values
    feature['feature_name'].append(col)
    feature['Number'].append(All[col].notna().sum())
    feature['Percent'].append(All[col].notna().sum()/len(All))
    for i in range(1, 5):
        feature[f'Sample{i}'].append(not_na_value[i-1])
        
pd.DataFrame(feature).to_csv('feature_info.csv', index=False)


# xml_paths = []
# with open('/data3/huyaojun/DrugTrail/data/all_xml', 'r') as f:
#     xml_paths = f.readlines()
#     xml_paths = ['/data3/huyaojun/DrugTrail/' + file.strip('\n') for file in xml_paths]
#     # print(xml_paths)

# disease2icd = load_disease2icd()
# print(disease2icd)

# All_data = pd.DataFrame()
# for i in range(1, 2):
#     print(f'Part {i}')
#     for path in tqdm(xml_paths[i*50000: (i+1)*50000]):
#         # if len(All_data) >50:
#         #     break
#         result = {}
#         # try:
#         if os.path.exists(path):
#             tree = ET.parse(path)
#             root = tree.getroot()
#             walkData_attrib(root, prefix = '', result = result)
#             # enroll = root.find('enrollment')
#             # if enroll:
#             #     enroll_attri = enroll.attrib
#             # else:
#             #     enroll_attri = np.nan
#             values = [strip_list(v) for v in result.values()]
#             # values.append(enroll_attri)
#             keys = list(result.keys())
#             # keys.append('enrollment_attrib')
#             if ('phase') in keys and ('4' in result['phase']):
#                 continue
#             if 'condition' in keys:
#                 icd_list = disease2icdcode(result['condition'], disease2icd)
#                 values.append(icd_list)
#                 keys.append('icdcode')
#                 # print('icdcode', icd_list)

#             ser = pd.Series(values, index=keys, name=str(result['id_info/nct_id'][0]))
            
#         else:
#             print(f'{path} is not exists!' )
#             ser = pd.Series(name=path[-15:-4])

#         All_data = pd.concat([All_data, pd.DataFrame(ser).T])
#     print(len(list(All_data.columns)))

#     # col_num = {}
#     # for col in All_data.columns:
#     #     col_num[col] = (~All_data[col].isna()).sum()
#     #     print(col, (~All_data[col].isna()).sum())

#     All_data.to_csv(f'All_data_part{i}.csv')
#     print(f'Save Part {i} !!!')
#     All_data = All_data.drop(index=All_data.index)

