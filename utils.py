import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import requests
import time
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec

## functions for analysis

def load_data():
    path = 'src/customDatabase.xlsx'
    df = pd.read_excel(path)
    with open('src/chemvecs_mol2vec.pkl', 'rb') as file:
        chemvecs_mol2vec = pickle.load(file)
        file.close()
    with open('src/chemvecs_morgan.pkl', 'rb') as file:
        chemvecs_morgan = pickle.load(file)
        file.close()
    return df, chemvecs_mol2vec, chemvecs_morgan


def dist_cosine(x, y):
    norm_x = np.sqrt((x * x).sum())
    norm_y = np.sqrt((y * y).sum())
    if 0 in [norm_x, norm_y]:
        return np.nan
    else:
        return max(1 - (x * y).sum() / (norm_x * norm_y), 0)


def dist_angle(x, y):
    norm_x = np.sqrt((x * x).sum())
    norm_y = np.sqrt((y * y).sum())
    if 0 in [norm_x, norm_y]:
        return np.nan
    else:
        arg = min(max((x * y).sum() / (norm_x * norm_y), -1), 1)
        return max(np.arccos(arg), 0)


def dist_euclidean(x, y):
    return max(np.sqrt(((x - y) ** 2).sum()), 0)


def dist_hamming(x, y):
    return len(set(x) | set(y)) - len(set(x) & set(y))


def dist_jaccard(x, y):
    return 1 - len(set(x) & set(y)) / len(set(x) | set(y))


def main_chemical_embedding(food_names, df_Custom, chemvecs_mol2vec, chemvecs_morgan, options):
    
    if options['chem']['use_full_database'] == 'True':
        add = [name for name in df_Custom['food_name'].unique() if name not in food_names]
        all_food_names = food_names + add
    else:
        all_food_names = food_names
    inds_to_render = []

    if ('t-SNE' in options['chem']['method']) and (options['chem']['distance_metric'] in ['hamming', 'jaccard']):
        X_food = []
    else:
        if options['chem']['vector_type'] == 'mol2vec':
            chemvecs = chemvecs_mol2vec
            dim = 300
        elif options['chem']['vector_type'] == 'morgan':
            chemvecs = chemvecs_morgan
            dim = 2048
        X_food = np.empty([len(all_food_names), dim])

    for i_food, food_name in enumerate(all_food_names):

        if food_name in food_names:
            inds_to_render.append(i_food)

        # slice dataframe
        df = df_Custom[df_Custom['food_name'] == food_name]

        # keep track of which compounds do and do not have documented taste or aroma
        is_chemosensory = []
        for i in range(len(df)):
            taste, aroma = df['tastes'].iloc[i], df['aromas'].iloc[i]
            no_taste, no_aroma = True, True
            if (type(taste) == str) and (taste != 'Tasteless'):
                no_taste = False
            if (type(aroma) == str) and (aroma != 'Odorless'):
                no_aroma = False
            if no_taste and no_aroma:
                is_chemosensory.append(False)
            else:
                is_chemosensory.append(True)
        is_chemosensory = np.array(is_chemosensory)

        if ('t-SNE' in options['chem']['method']) and (options['chem']['distance_metric'] in ['hamming', 'jaccard']):

            # the food is just the list of its compounds
            # structures and amounts are irrelevant because distances between foods are simply the numbers of compounds not in common
            if options['chem']['no_chemosensory_policy'] == 'include':
                X_food.append([name for name in df['compound_name']])
            else:
                X_food.append([name for i_name, name in enumerate(df['compound_name']) if is_chemosensory[i_name]])

        else:

            # keep track of which compounds do and do not have documented chemical structures
            has_structure = np.array([key in chemvecs for key in df['compound_name']])
            # convert chemical structures to vectors
            X_compounds = np.array([(chemvecs[key] if key in chemvecs else np.zeros(dim)) for key in df['compound_name']])

            # take a relative quantity-weighted sum of the compound vectors to obtain a food vector
            # sum
            if options['chem']['no_chemosensory_policy'] != 'include':
                X_compounds[~is_chemosensory, :] = 0
            a = np.array(df['amount'])
            x_food = (a.reshape([1, -1]) @ X_compounds).flatten()
            # normalize
            if options['chem']['no_structure_policy'] == 'remove':
                a[~has_structure] = 0
            if options['chem']['no_chemosensory_policy'] == 'remove':
                a[~is_chemosensory] = 0
            den = a.sum()
            X_food[i_food, :] = x_food / den

    # optionally normalize data matrix
    if (options['chem']['distance_metric'] not in ['hamming', 'jaccard']) and options['chem']['z_score']:
        mu = X_food.mean(axis=0).reshape([1, -1])
        sigma = X_food.std(axis=0)
        sigma = np.array([(sigma_ if sigma_ != 0 else 1) for sigma_ in sigma]).reshape([1, -1])
        X_food = (X_food - mu) / sigma

    if ('PCA' in options['chem']['method']):

        # PCA
        n_components = int(options['chem']['method'][0])
        X_demean = X_food - X_food.mean(axis=0).reshape([1, -1])
        C = X_demean.T @ X_demean / (X_demean.shape[0] - 1)
        pca = PCA(n_components=n_components)
        score = pca.fit_transform(X_demean)

        return C, score[:, :n_components], inds_to_render
        
    elif ('t-SNE' in options['chem']['method']):

        # distance calculations
        if options['chem']['distance_metric'] == 'cosine':
            dist = lambda x, y: dist_cosine(x, y)
        elif options['chem']['distance_metric'] == 'angle':
            dist = lambda x, y: dist_angle(x, y)
        elif options['chem']['distance_metric'] == 'euclidean':
            dist = lambda x, y: dist_euclidean(x, y)
        elif options['chem']['distance_metric'] == 'hamming':
            dist = lambda x, y: dist_hamming(x, y)
        elif options['chem']['distance_metric'] == 'jaccard':
            dist = lambda x, y: dist_jaccard(x, y)

        if options['chem']['distance_metric'] in ['hamming', 'jaccard']:
            pdist = np.zeros([len(X_food), len(X_food)])
            for i in range(len(X_food)):
                for j in range(len(X_food)):
                    pdist[i, j] = dist(X_food[i], X_food[j])
        else:
            pdist = np.zeros([X_food.shape[0], X_food.shape[0]])
            for i in range(X_food.shape[0]):
                for j in range(X_food.shape[0]):
                    pdist[i, j] = dist(X_food[i, :], X_food[j, :])

        # t-SNE
        n_components = int(options['chem']['method'][0])
        np.random.seed(9999995)
        if options['chem']['distance_metric'] in ['hamming', 'jaccard', 'angle']:
            tsne = TSNE(metric='precomputed', n_components=n_components, perplexity=options['chem']['perplexity'], init='random')
            X_tsne = tsne.fit_transform(pdist)
        else:
            tsne = TSNE(metric=options['chem']['distance_metric'], n_components=n_components, perplexity=options['chem']['perplexity'])
            X_tsne = tsne.fit_transform(X_food)
        
        return pdist, X_tsne, inds_to_render


def get_taste_vector(food_name, df_Custom, no_chemosensory_policy):

    tastes = ['Bitter', 'Pungent', 'Astringent', 'Sweet', 'Sour', 'Cool', 'Umami', 'Salty']

    df = df_Custom[df_Custom['food_name'] == food_name]

    # convert compounds to taste vectors
    X_compounds = np.zeros([len(df), len(tastes)])
    for i_compound in range(len(df)):
        compound_taste_vector = np.zeros(len(tastes))
        compound_tastes = df['tastes'].iloc[i_compound]
        if type(compound_tastes) == str:
            compound_taste_list = compound_tastes.split(', ')
            for taste in compound_taste_list:
                compound_taste_vector += np.array([float(taste == taste_) for taste_ in tastes])
        X_compounds[i_compound, :] = compound_taste_vector

    # keep track of the compounds that lack taste data
    is_chemosensory = np.array([any(X_compounds[i_compound, :] != 0) for i_compound in range(X_compounds.shape[0])])

    # take a relative quantity-weighted sum of the compound taste vectors to obtain a food taste vector
    # sum
    a = np.array(df['amount'])
    x_food = (a.reshape([1, -1]) @ X_compounds).flatten()
    # normalize
    if no_chemosensory_policy == 'remove':
        a[~is_chemosensory] = 0
    den = a.sum()
    x_food /= den

    return x_food, tastes

def main_taste_embedding(food_names, df_Custom, options):

    if options['taste']['use_full_database'] == 'True':
        add = [name for name in df_Custom['food_name'].unique() if name not in food_names]
        all_food_names = food_names + add
    else:
        all_food_names = food_names
    inds_to_render = []

    tastes = ['Bitter', 'Pungent', 'Astringent', 'Sweet', 'Sour', 'Cool', 'Umami', 'Salty']

    X_food = np.empty([len(all_food_names), len(tastes)])

    for i_food, food_name in enumerate(all_food_names):

        if food_name in food_names:
            inds_to_render.append(i_food)

        df = df_Custom[df_Custom['food_name'] == food_name]

        # convert compounds to taste vectors
        X_compounds = np.zeros([len(df), len(tastes)])
        for i_compound in range(len(df)):
            compound_taste_vector = np.zeros(len(tastes))
            compound_tastes = df['tastes'].iloc[i_compound]
            if type(compound_tastes) == str:
                compound_taste_list = compound_tastes.split(', ')
                for taste in compound_taste_list:
                    compound_taste_vector += np.array([float(taste == taste_) for taste_ in tastes])
            X_compounds[i_compound, :] = compound_taste_vector

        # keep track of the compounds that lack taste data
        is_chemosensory = np.array([any(X_compounds[i_compound, :] != 0) for i_compound in range(X_compounds.shape[0])])

        # take a relative quantity-weighted sum of the compound taste vectors to obtain a food taste vector
        # sum
        a = np.array(df['amount'])
        x_food = (a.reshape([1, -1]) @ X_compounds).flatten()
        # normalize
        if options['taste']['no_chemosensory_policy'] == 'remove':
            a[~is_chemosensory] = 0
        den = a.sum()
        X_food[i_food, :] = x_food / den

    # optionally normalize data matrix
    if options['taste']['z_score']:
        mu = X_food.mean(axis=0).reshape([1, -1])
        sigma = X_food.std(axis=0)
        sigma = np.array([(sigma_ if sigma_ != 0 else 1) for sigma_ in sigma]).reshape([1, -1])
        X_food = (X_food - mu) / sigma

    if ('PCA' in options['taste']['method']):

        # PCA
        n_components = int(options['taste']['method'][0])
        X_demean = X_food - X_food.mean(axis=0).reshape([1, -1])
        C = X_demean.T @ X_demean / (X_demean.shape[0] - 1)
        pca = PCA(n_components=n_components)
        score = pca.fit_transform(X_demean)

        return C, score[:, :n_components]

    elif ('t-SNE' in options['taste']['method']):

        # distance calculations
        if options['taste']['distance_metric'] == 'cosine':
            dist = lambda x, y: dist_cosine(x, y)
        elif options['taste']['distance_metric'] == 'angle':
            dist = lambda x, y: dist_angle(x, y)
        elif options['taste']['distance_metric'] == 'euclidean':
            dist = lambda x, y: dist_euclidean(x, y)

        pdist = np.zeros([X_food.shape[0], X_food.shape[0]])
        for i in range(X_food.shape[0]):
            for j in range(X_food.shape[0]):
                pdist[i, j] = dist(X_food[i, :], X_food[j, :])

        n_components = int(options['taste']['method'][0])
        np.random.seed(9999995)
        if options['taste']['distance_metric'] == 'angle':
            tsne = TSNE(metric='precomputed', n_components=n_components, perplexity=options['taste']['perplexity'], init='random')
            X_tsne = tsne.fit_transform(pdist)
        else:
            tsne = TSNE(metric=options['taste']['distance_metric'], n_components=n_components, perplexity=options['taste']['perplexity'])
            X_tsne = tsne.fit_transform(X_food)

        return pdist, X_tsne, inds_to_render


## functions for updating database

def is_valid(val):
    if val is None:
        return False
    missing_vals = ['None', 'none', 'NaN', 'nan', np.nan, np.NaN, np.NAN, '', ' ']
    if val in missing_vals:
        return False
    return True

def updated_sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """

    keys = set(model.wv.key_to_index)
    vec = []

    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence 
                            if y in set(sentence) & keys]))

    return np.array(vec)

def get_chemical_compounds(search_name, KEY, quantified=True, verbose=False):
    '''
    food --> compounds
    
    Takes a food search name and uses the FooDB API to return a pandas dataframe 
    with a row for each quantified compound in the food
    It also returns the name of the food, which may be different from the search name
    (if it is, a warning will be printed)
    '''

    '''
    food --> compounds
    
    Takes a food search name and uses the FooDB API to return a pandas dataframe 
    with a row for each quantified compound in the food
    It also returns the name of the food, which may be different from the search name
    (if it is, a warning will be printed)
    '''

    # get the food id
    response = requests.post("http://35.184.189.38/api/foodb/food/search/?api-key={}".format(KEY), json={"search_term": search_name})
    if response.status_code != 200:
        print('API FAILURE')
        return (None, None)
    results = response.json()['results']
    if len(results) > 0:
        match = [result['name'] == search_name for result in results]
        if sum(match) > 0:
            result = results[np.argmax(match)]
        else:
            result = results[0]
            print("Inexact match for search term '{}' ... changed to '{}'".format(search_name, result['name']))
        food_name, food_id = result['name'], result['public_id']
    else:
        print("No results for search term '{}'".format(search_name))
        return (None, None)

    # get the compounds
    response = requests.get("http://35.184.189.38/api/foodb/food/{}/compounds/?api-key={}".format(food_id, KEY))
    if response.status_code != 200:
        print('API FAILURE')
        return (None, None)
    if verbose:
        print('Total number of compounds:', response.json()['num_compounds'])
    results = response.json()['data']
    compound_names = [result['name'] for result in results]
    compound_amounts = [result['average_content'] for result in results]
    for i_amount, amount in enumerate(compound_amounts):
        if amount is None:
            compound_amounts[i_amount] = 0
        elif type(amount) == str:
            compound_amounts[i_amount] = float(amount.split()[0])
    compound_amounts_relative = [am / sum(compound_amounts) for am in compound_amounts]
    compound_ids = [result['primary_id'] for result in results]
    compound_ids_cas = [result['cas_number'] for result in results]
    compound_ids_hmdb = [result['external_links']['hmdb_id'] for result in results]
    compound_ids_pubchem = [result['external_links']['pubchem_compound_id'] for result in results]
    compound_ids_flavornet = [result['external_links']['flavornet_id'] for result in results]

    if food_name == 'Strawberry':
        # supplement
        path = 'C:/Users/Liam/Downloads/Strawberry.xlsx'
        df_strawberry = pd.read_excel(path)
        cas_ids = [str(df_strawberry['CAS Number'].iloc[i]) for i in range(1, len(df_strawberry))]
        for i in range(len(cas_ids)):
            if '"' in cas_ids[i]:
                cas_ids[i] = cas_ids[i][1:-1]

        pubchem_ids = [str(int(df_strawberry['PubChem ID'].iloc[i])) for i in range(1, len(df_strawberry))]

        names = [df_strawberry['Compounds'].iloc[i] for i in range(1, len(df_strawberry))]
        content = []
        for i in range(1, len(df_strawberry)):
            conts = [df_strawberry.iloc[i, j] if type(df_strawberry.iloc[i, j]) == float else 0 for j in range(4, 20)]
            content.append(np.mean(conts) * 1e-4)
        compound_names += names
        compound_amounts += content
        compound_amounts_relative = [am / sum(compound_amounts) for am in compound_amounts]
        compound_ids += [None for i in range(len(names))]
        compound_ids_cas += cas_ids
        compound_ids_hmdb += [None for i in range(len(names))]
        compound_ids_pubchem += pubchem_ids
        compound_ids_flavornet += [None for i in range(len(names))]

    df = pd.DataFrame({
        'food_name': [food_name] * len(compound_names),
        'compound_name': compound_names,
        'amount': compound_amounts,
        'rel_amount': compound_amounts_relative,
        'id': compound_ids,
        'id_cas': compound_ids_cas,
        'id_hmdb': compound_ids_hmdb,
        'id_pubchem': compound_ids_pubchem,
        'id_flavornet': compound_ids_flavornet
    })
    df = df.sort_values(by='amount', ascending=False)
    if quantified:
        df = df[df['amount'] > 0]
        if verbose:
            print('Number of{}compounds: {}'.format((' quantified ' if quantified else ' '), len(df)))

    return (food_name, df)


def get_aromas(df_, df_Aroma, verbose=False):
    '''
    compounds --> aromas
    
    Takes a dataframe with a row for each compound and returns a new dataframe
    with aromas for each (uses the local master Aroma database)
    '''

    df = df_.copy(deep=True)

    smiles_all = [df_Aroma['SMILES'].iloc[i] for i in range(len(df_Aroma))]
    pubchem_ids_all = [df_Aroma['PubChemID'].iloc[i] for i in range(len(df_Aroma))]
    names_all = [str(df_Aroma['Name'].iloc[i]).lower() for i in range(len(df_Aroma))]
    cas_ids_all = [df_Aroma['CAS_ID'].iloc[i] for i in range(len(df_Aroma))]
    aromas = []
    for i in range(len(df)):
        smiles = df['smiles'].iloc[i]
        pubchem_id = df['id_pubchem'].iloc[i]
        name = df['compound_name'].iloc[i].lower()
        cas_id = df['id_cas'].iloc[i]
        if is_valid(smiles) and smiles in smiles_all:
            filt = [smiles_ == smiles for smiles_ in smiles_all]
            aromas.append(df_Aroma[filt]['Flavor type'].iloc[0])
        elif is_valid(pubchem_id) and pubchem_id in pubchem_ids_all:
            filt = [id_ == pubchem_id for id_ in pubchem_ids_all]
            aromas.append(df_Aroma[filt]['Flavor type'].iloc[0])
        elif is_valid(cas_id) and cas_id in cas_ids_all:
            filt = [id_ == cas_id for id_ in cas_ids_all]
            aromas.append(df_Aroma[filt]['Flavor type'].iloc[0])
        elif is_valid(name) and name in names_all:
            filt = [name_ == name for name_ in names_all]
            aromas.append(df_Aroma[filt]['Flavor type'].iloc[0])
        else:
            aromas.append(None)
    df['aromas'] = aromas

    return df


def get_tastes(df_, df_Taste, verbose=False):
    '''
    compounds --> tastes
    
    Takes a dataframe with a row for each compound and returns a new dataframe
    with tastes for each (uses the local master Taste database)
    '''

    df = df_.copy(deep=True)

    smiles_all = [df_Taste['SMILES'].iloc[i] for i in range(len(df_Taste))]
    pubchem_ids_all = [df_Taste['PubChemID'].iloc[i] for i in range(len(df_Taste))]
    names_all = [str(df_Taste['Name'].iloc[i]).lower() for i in range(len(df_Taste))]
    cas_ids_all = [df_Taste['CAS_ID'].iloc[i] for i in range(len(df_Taste))]
    tastes = []
    for i in range(len(df)):
        smiles = df['smiles'].iloc[i]
        pubchem_id = df['id_pubchem'].iloc[i]
        name = df['compound_name'].iloc[i].lower()
        cas_id = df['id_cas'].iloc[i]
        if is_valid(smiles) and smiles in smiles_all:
            filt = [smiles_ == smiles for smiles_ in smiles_all]
            tastes.append(df_Taste[filt]['Flavor type'].iloc[0])
        elif is_valid(pubchem_id) and pubchem_id in pubchem_ids_all:
            filt = [id_ == pubchem_id for id_ in pubchem_ids_all]
            tastes.append(df_Taste[filt]['Flavor type'].iloc[0])
        elif is_valid(cas_id) and cas_id in cas_ids_all:
            filt = [id_ == cas_id for id_ in cas_ids_all]
            tastes.append(df_Taste[filt]['Flavor type'].iloc[0])
        elif is_valid(name) and name in names_all:
            filt = [name_ == name for name_ in names_all]
            tastes.append(df_Taste[filt]['Flavor type'].iloc[0])
        else:
            tastes.append(None)
    df['tastes'] = tastes

    return df


def get_chemical_structures(df_, search_without_id=False, verbose=False):
    '''
    compounds --> structures
    
    Takes a dataframe with a row for each compound and returns a new dataframe
    with a SMILES structure for each (uses the PubChem API to get the structure)
    '''

    df = df_.copy(deep=True)

    # the best way to get the structures is through PubChem using the compound's PubChem ID (because multiple
    # compounds can be processed this way at once)
    # however, not every compound has this ID easily available - they have to be dealt with later
    
    # get the structures with PubChem IDs if possible
    pubchem_ids = [df['id_pubchem'].iloc[i] for i in range(len(df))]
    has_pubchem_id = [is_valid(item) for item in pubchem_ids]
    n_chunks = int(np.ceil(sum(has_pubchem_id) / 100))
    compound_smiles_ = []
    list_of_ids = [ df[has_pubchem_id]['id_pubchem'].iloc[i] for i in range(sum(has_pubchem_id)) ]
    for chunk in range(n_chunks):
        response = requests.get(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/CanonicalSMILES/json'
            .format(','.join(list_of_ids[(100 * chunk):min(100 * (chunk + 1), len(list_of_ids))])))
        if response.status_code != 200:
            print('API FAILURE')
            print(response.status_code)
            raise Exception('PUBCHEM API FAILURE (ids --> [SMILES])')
        compound_smiles_ += [x['CanonicalSMILES'] for x in response.json()['PropertyTable']['Properties']]
        time.sleep(0.2)

    # build full set of structures with placeholders
    compound_smiles = []
    count = 0
    for i in range(len(has_pubchem_id)):
        if has_pubchem_id[i]:
            compound_smiles.append(compound_smiles_[count])
            count += 1
        else:
            if search_without_id:
                response = requests.get(
                    'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/json'
                    .format(df['compound_name'].iloc[i]))
                if response.status_code != 200:
                    compound_smiles.append(None)
                else:
                    compound_smiles.append(response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES'])
                    pubchem_ids[i] = response.json()['PropertyTable']['Properties'][0]['CID']
                time.sleep(0.2)
            else:
                compound_smiles.append(None)

    df['smiles'] = compound_smiles
    df['id_pubchem'] = pubchem_ids
    if verbose:
        n_struct = sum([is_valid(df['smiles'].iloc[i]) for i in range(len(df))])
        print('Number of compounds with structure:', n_struct)
        print('\nFirst compound (by composition):\n')

    return df


def data_to_embedding(df_, model):
    '''
    structures --> vectors
    
    Takes a dataframe with a row for each compound and uses mol2vec to convert each
    SMILES structure into a vector
    Returns a new dataframe with the vectors in a new column and also a numpy array
    of shape (number_of_compounds, dimensionality_of_vector)
    '''

    df = df_.copy(deep=True)

    # transform SMILES to MOL
    df['mol'] = df['smiles'].apply(lambda x: (Chem.MolFromSmiles(x) if is_valid(x) else None))

    # construct sentences
    df['sentence'] = df.apply(lambda x: (MolSentence(mol2alt_sentence(x['mol'], 1)) if x['mol'] is not None else None), axis=1)

    # extract embeddings to a numpy array
    # note that we always should mark unseen='UNK' in sentence2vec() 
    #   so that model is taught how to handle unknown substructures
    has_struct = [x is not None for x in df['sentence']]
    sentences = updated_sentences2vec(df[has_struct]['sentence'], model, unseen='UNK')
    mol2vec, count = [], 0
    for i in range(len(has_struct)):
        if has_struct[i]:
            mol2vec.append(sentences[count])
            count += 1
        else:
            mol2vec.append(None)
    df['mol2vec'] = [(DfVec(x) if x is not None else None) for x in mol2vec]
    X = np.array([(x.vec if x is not None else np.zeros(300)) for x in df['mol2vec']])

    return (X, df)


def data_to_embedding2(df_, model):
    '''
    structures --> vectors
    
    Takes a dataframe with a row for each compound and uses Morgan generator to convert each
    SMILES structure into a vector
    Returns a new dataframe with the vectors in a new column and also a numpy array
    of shape (number_of_compounds, dimensionality_of_vector)
    '''

    df = df_.copy(deep=True)

    # transform SMILES to MOL
    df['mol'] = df['smiles'].apply(lambda x: (Chem.MolFromSmiles(x) if is_valid(x) else None))

    # construct vectors
    morgan_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    df['fp'] = df.apply(lambda x: (morgan_generator.GetFingerprintAsNumPy(x['mol']) if x['mol'] is not None else None), axis=1)

    # extract embeddings to a numpy array
    X = np.array([(x if x is not None else np.zeros(2048)) for x in df['fp']])

    return (X, df)