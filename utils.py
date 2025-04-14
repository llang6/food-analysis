import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_data():
    path = 'customDatabase.xlsx'
    df = pd.read_excel(path)
    with open('chemvecs_mol2vec.pkl', 'rb') as file:
        chemvecs_mol2vec = pickle.load(file)
        file.close()
    with open('chemvecs_morgan.pkl', 'rb') as file:
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
    
    if ('t-SNE' in options['chem']['method']) and (options['chem']['distance_metric'] in ['hamming', 'jaccard']):
        X_food = []
    else:
        if options['chem']['vector_type'] == 'mol2vec':
            chemvecs = chemvecs_mol2vec
            dim = 300
        elif options['chem']['vector_type'] == 'morgan':
            chemvecs = chemvecs_morgan
            dim = 2048
        X_food = np.empty([len(food_names), dim])

    for i_food, food_name in enumerate(food_names):

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

        return C, score[:, :n_components]
        
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
        
        return pdist, X_tsne
    

def main_taste_embedding(food_names, df_Custom, options):

    tastes = ['Bitter', 'Pungent', 'Astringent', 'Sweet', 'Sour', 'Cool', 'Umami', 'Salty']

    X_food = np.empty([len(food_names), len(tastes)])

    for i_food, food_name in enumerate(food_names):

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
        if options['chem']['distance_metric'] == 'cosine':
            dist = lambda x, y: dist_cosine(x, y)
        elif options['chem']['distance_metric'] == 'angle':
            dist = lambda x, y: dist_angle(x, y)
        elif options['chem']['distance_metric'] == 'euclidean':
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

        return pdist, X_tsne
