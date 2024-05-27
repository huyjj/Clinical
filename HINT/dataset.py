

# import torch, os, sys
# torch.manual_seed(0) 
# sys.path.append('.')
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
from HINT.dataloader import Trial_Dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
# from HINT.protocol_encode import protocol2feature, sentence2vec
# from HINT.utils import smiles_txt_to_lst, icdcode_text_2_lst_of_lst
from HINT.text_encode import text2feature
from HINT.dataloader import smiles_txt_to_lst, protocol2feature, icdcode_text_2_lst_of_lst, sentence2vec
from HINT.mesh_encode import mesh_term2feature

class Trial_Dataset_tabular(Trial_Dataset):
    def __init__(self, nctid_lst, label_lst, icdcode_lst, smiles_lst, criteria_lst, tabular_lst, text_lst, mesh_lst):
        Trial_Dataset.__init__(self, nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst)
        self.tabular_lst = tabular_lst
        self.text_lst = text_lst
        self.mesh_lst = mesh_lst

    def __getitem__(self, index):
        return self.nctid_lst[index], self.label_lst[index], self.smiles_lst[index], self.icdcode_lst[index], self.criteria_lst[index], self.tabular_lst[index], self.text_lst[index], self.mesh_lst[index]


def refine_year(x):
        # change to Months
        if type(x) == str:
            if 'Year' in x:
                number = eval(x.split(' ')[0])       
                return number * 12
            elif 'Month' in x:
                number = eval(x.split(' ')[0])
                return number
            elif 'Week' in x:
                number = eval(x.split(' ')[0])
                return number / 4.286
            elif 'Day' in x:
                number = eval(x.split(' ')[0])
                return number / 30
            elif 'Hour' in x:
                number = eval(x.split(' ')[0])
                return number / 30 / 24
            elif 'Minute' in x:
                number = eval(x.split(' ')[0])
                return number / 30 / 24 / 60
        return x


def trial_tabular_collate_fn(x):
    nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
    label_vec = default_collate([i[1] for i in x])  ### shape n, 
    smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
    criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
    tabular_lst = [i[5] for i in x]
    text_lst = [text2feature(i[6]) for i in x]
    mesh_lst = [mesh_term2feature(i[7]) for i in x]
    return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst, tabular_lst, text_lst, mesh_lst]

def trial_tabular_2_collate_fn(x):
    nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
    label_vec = torch.tensor([i[1] for i in x])  ### shape n, 2
    smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
    criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
    tabular_lst = [i[5] for i in x]
    text_lst = [text2feature(i[6]) for i in x]
    mesh_lst = [mesh_term2feature(i[7]) for i in x]
    return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst, tabular_lst, text_lst, mesh_lst]


def remove_unused_column(data):
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data

def split_data(data, target, test_size):
    label = data[target]
    data = data.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=123, shuffle=True)
    return X_train, y_train.values, X_test, y_test.values


def quantile_transform(X_train, X_valid, X_test):
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_valid = qt.transform(X_valid)
    X_test = qt.transform(X_test)

    return X_train, X_valid, X_test

def mortality_rate(phase):
    target = 'mortality_rate'
    X_train = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_train_x.csv')
    y_train = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_train_y.csv')[['mortality_rate', 'Y/N']]
    X_test = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_test_x.csv')
    y_test = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_test_y.csv')[['mortality_rate', 'Y/N']]

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train.loc[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test.loc[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def mortality_rate_yn(phase):
    target = 'mortality_rate'
    X_train = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_train_x.csv')
    y_train = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_train_y.csv')['Y/N']
    X_test = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_test_x.csv')
    y_test = pd.read_csv('../adverse-event-rate-prediction/mortality_rate_test_y.csv')['Y/N']

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)


def serious_adverse_rate(phase):
    
    target = 'serious_adverse_rate'
    X_train = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_train_x.csv')
    y_train = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_train_y.csv')[['serious_adverse_rate', 'Y/N']]
    X_test = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_test_x.csv')
    y_test = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_test_y.csv')[['serious_adverse_rate', 'Y/N']]
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def serious_adverse_rate_yn(phase):
    target = 'serious_adverse_rate'
    X_train = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_train_x.csv')
    y_train = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_train_y.csv')['Y/N']
    X_test = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_test_x.csv')
    y_test = pd.read_csv('../adverse-event-rate-prediction/serious_adverse_rate_test_y.csv')['Y/N']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)

def patient_dropout_rate(phase):
    target = 'patient_dropout_rate'
    X_train = pd.read_csv('../patient-dropout-prediction/train_x.csv')
    y_train = pd.read_csv('../patient-dropout-prediction/train_y.csv')[['droupout_rate', 'Y/N']]
    X_test = pd.read_csv('../patient-dropout-prediction/test_x.csv')
    y_test = pd.read_csv('../patient-dropout-prediction/test_y.csv')[['droupout_rate', 'Y/N']]
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def patient_dropout_rate_yn(phase):
    target = 'patient_dropout_rate'
    X_train = pd.read_csv('../patient-dropout-prediction/train_x.csv')
    y_train = pd.read_csv('../patient-dropout-prediction/train_y.csv')['Y/N']
    X_test = pd.read_csv('../patient-dropout-prediction/test_x.csv')
    y_test = pd.read_csv('../patient-dropout-prediction/test_y.csv')['Y/N']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)



def duration(phase):
    X_train = pd.read_csv('../trial-duration-prediction/train_x_2022.csv')
    y_train = pd.read_csv('../trial-duration-prediction/train_y_2022.csv')['time_day']
    X_test = pd.read_csv('../trial-duration-prediction/test_x_2022.csv')
    y_test = pd.read_csv('../trial-duration-prediction/test_y_2022.csv')['time_day']
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # maxmin scalar
    # min_val = 0
    # max_val = 10
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def outcome(phase):
    target = 'outcome'
    X_train = pd.read_csv('../trial-approval-prediction/train_x.csv')
    y_train = pd.read_csv('../trial-approval-prediction/train_y.csv')['outcome']
    X_test = pd.read_csv('../trial-approval-prediction/test_x.csv')
    y_test = pd.read_csv('../trial-approval-prediction/test_y.csv')['outcome']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)

def failure_reason(phase):
    def mapper(x):
        if x == 'poor enrollment':
            return 0
        elif x == 'efficacy':
            return 1
        elif x == 'safety':
            return 2
        elif x == 'Others':
            return 3
    target = 'failure_reason'
    X_train = pd.read_csv('../trial-failure-reason-prediction/train_x.csv')
    y_train = pd.read_csv('../trial-failure-reason-prediction/train_y.csv')['failure_reason'].apply(mapper)
    X_test = pd.read_csv('../trial-failure-reason-prediction/test_x.csv')
    y_test = pd.read_csv('../trial-failure-reason-prediction/test_y.csv')['failure_reason'].apply(mapper)
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 4, len(numerical_features + hot_features)


class Dose_dataset(Dataset):
    def __init__(self, nctid_lst, label_lst, smiles_lst,  mesh_lst):
        self.nctid_lst = nctid_lst
        self.label_lst = label_lst
        self.smiles_lst = smiles_lst
        self.mesh_lst = mesh_lst
    
    def __len__(self):
        return len(self.nctid_lst)

    def __getitem__(self, idx):
        return self.nctid_lst[idx], self.label_lst[idx], self.smiles_lst[idx], self.mesh_lst[idx]

# def dose_collate_fn(x):
#     nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
#     label_vec = default_collate([i[1] for i in x])  ### shape n, 
#     smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
#     mesh_lst = [mesh_term2feature(i[3]) for i in x]
#     return [nctid_lst, label_vec, smiles_lst, mesh_lst]

def dose(phase):
    X_train = pd.read_csv('../drug-dose-prediction/train_x.csv')
    y_train = pd.read_csv('../drug-dose-prediction/train_y.csv')[['Max','Min']]
    X_test = pd.read_csv('../drug-dose-prediction/test_x.csv')
    y_test = pd.read_csv('../drug-dose-prediction/test_y.csv')[['Max','Min']]
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    numerical_features = []
    text_features = ['smiless']
    hot_features = []
    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    def dose_collate_fn(x):
        nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
        label_vec = default_collate([np.array(i[1]) for i in x])  ### shape n, 
        smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
        mesh_lst = [mesh_term2feature(i[3]) for i in x]
        return [nctid_lst, label_vec, smiles_lst, mesh_lst]

    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.values.tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Dose_dataset(train_nctid_lst, train_label_lst, train_smiles_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dose_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Dose_dataset(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.values.tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Dose_dataset(test_nctid_lst, test_label_lst, test_smiles_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    return train_loader, valid_loader, test_loader, 2, 0



def dose_cls(phase):
    X_train = pd.read_csv('../drug-dose-prediction/train_x.csv')
    y_train = pd.read_csv('../drug-dose-prediction/train_y_cls.csv')['Avg']
    X_test = pd.read_csv('../drug-dose-prediction/test_x.csv')
    y_test = pd.read_csv('../drug-dose-prediction/test_y_cls.csv')['Avg']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    numerical_features = []
    text_features = ['smiless']
    hot_features = []
    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]
    
    def dose_collate_fn(x):
        nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
        label_vec = default_collate([i[1] for i in x])  ### shape n, 
        smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
        mesh_lst = [mesh_term2feature(i[3]) for i in x]
        return [nctid_lst, label_vec, smiles_lst, mesh_lst]

    train_nctid_lst = X_train['nctid'].tolist()
    train_label_lst = y_train.tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Dose_dataset(train_nctid_lst, train_label_lst, train_smiles_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dose_collate_fn)
    
    valid_nctid_lst = X_valid['nctid'].tolist()
    valid_label_lst = y_valid.tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Dose_dataset(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    test_nctid_lst = X_test['nctid'].tolist()
    test_label_lst = y_test.tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Dose_dataset(test_nctid_lst, test_label_lst, test_smiles_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    return train_loader, valid_loader, test_loader, 4, 0


class Repurpose(Dataset):
    def __init__(self, icdcode_lst, disease_mesh_lst, smiles_lst, drug_mesh_lst):
        super(Repurpose, self).__init__()
        self.icdcode_lst = icdcode_lst
        self.disease_mesh_lst = disease_mesh_lst
        self.smiles_lst = smiles_lst
        self.drug_mesh_lst = drug_mesh_lst
    
    def __len__(self):
        return len(self.icdcode_lst)

    def __getitem__(self, index):
        return self.icdcode_lst[index], self.disease_mesh_lst[index], self.smiles_lst[index], self.drug_mesh_lst[index]


def Repurpose_collate_fn(x):
    smiless = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode = [icdcode_text_2_lst_of_lst(i[0]) for i in x]
    drug_term_embed = [mesh_term2feature(i[-1]) for i in x]
    disease_term_embed = [mesh_term2feature(i[1]) for i in x]
    disease_term = [i[1] for i in x]
    drug_term = [i[-1] for i in x]
    label = [torch.tensor(1.) for i in range(len(x))]
    return [disease_term_embed, icdcode, drug_term_embed, smiless, disease_term, drug_term, label]

def repurposing():
    train_part = pd.read_csv('../drug-repurposing/train_part.csv', index_col=0)
    X_test = pd.read_csv('../drug-repurposing/test_2018.csv', index_col=0)
    import json
    with open('../drug-repurposing/drug_mesh_term2smiless.json', 'r') as f:
        drug_mesh_term2smiless = json.load(f)
    
    def smiless2(x):
        if len(x) == 1 and len(x[0]) < 1:
            return "['unknown']"
        elif x == ['']:
            return "['unknown']"
        elif x == "['']":
            return "['unknown']"
        elif x == "[]" :
            return "['unknown']"
        else:
            return x
    train_icdcode_lst = train_part['icdcodes'].fillna('["unknown"]').tolist()
    train_smiles_lst = train_part['smiless'].fillna('["unknown"]')
    train_smiles_lst  = train_smiles_lst.apply(smiless2).tolist()
    train_drug_mesh_lst = train_part['intervention_browse/mesh_term'].values.tolist()
    train_disease_mesh_lst = train_part['condition_browse/mesh_term'].values.tolist()
    train_dataset = Repurpose(train_icdcode_lst, train_disease_mesh_lst, train_smiles_lst, train_drug_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=Repurpose_collate_fn)

    test_icdcode_lst = X_test['icdcodes'].fillna('["unknown"]').tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]')
    test_smiles_lst = test_smiles_lst.apply(smiless2).tolist()
    test_drug_mesh_lst = X_test['intervention_browse/mesh_term'].values.tolist()
    test_disease_mesh_lst = X_test['condition_browse/mesh_term'].values.tolist()
    test_dataset = Repurpose(test_icdcode_lst, test_disease_mesh_lst, test_smiles_lst, test_drug_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=Repurpose_collate_fn)
    
    return train_loader, train_loader, test_loader, 0, 0



if __name__ == '__main__':
    # # X_train, y_train, X_valid, y_valid, X_test, y_test, text_features, hot_features, numerical_features, mesh_term = dose()
    # # print(y_train)
    # train_pairs, valid_pairs, test_pairs = repurposing()
    repurposing()
    # print(train_pairs)
    