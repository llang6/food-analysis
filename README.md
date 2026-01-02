Dash app hosted on Posit Connect Cloud here:
https://llang6-food-analysis.share.connect.posit.cloud

**Database**

- The list of quantified compounds for each food comes from _FooDB_
- The chemical structure (SMILES) for each compound comes from _PubChem_
- The 300-dimensional chemical embeddings come from a pre-trained _mol2vec_ model (<https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl>)
- The 2048-dimensional chemical embeddings come from the Morgan generator in _rdkit_
- Taste information for each compound comes from a meta-database published as _Data-driven Elucidation of Flavor Chemistry_ (<https://github.com/ecological-systems-design/flavor-chemical-design/blob/main/5_excel/Taste_molecules_8982.xls>)
- There is also aroma information for each compound in this repository

**Visualization**

- The variable `food_and_colors` is for manually setting which foods you want to be displayed in plots and what their dot colors should be

- The set of all foods upon which a "distance space" (or "covariance space") will be constructed will either be this manually-defined set of foods or all foods in the database (if `use_full_database` is true)

**Visualization in chemical space**

- In the database, each food, $\mathrm{food}_i$, is composed of a list of $N_i$ compounds along with their amounts, whether or not they have a documented chemical structure, and whether or not they have a documented chemosensory quality (taste _or_ aroma):

$$
\mathrm{food}_i \leftrightarrow [ \mathrm{compound}_1, ..., \mathrm{compound}_j, ..., \mathrm{compound}_{N_i} ]
$$

$$
\mathrm{compound}_j \leftrightarrow (\mathrm{name}_j, a_j, \delta_j^\mathrm{struct}, \delta_j^\mathrm{chemo})
$$

- The chemical representation of each food, $F_i$, is a function of its compounds

- If you choose to visualize the foods in a distance space using either the Hamming or Jaccard metrics, each food will be:

    - A set of compounds
    
$\mathbf{f}_i^\mathrm{C} = \{ \mathrm{name}_1, ..., \mathrm{name}_j, ..., \mathrm{name}_{N_i} \}$

    - Otherwise, each food will be:
    
        - A weighted sum of its compound vectors, each of which is either a 300-dimensional _mol2vec_ embedding or a 2048-dimensional _morgan_ embedding (choose with the `vector_type` variable)
        
        - If a compound's structure is not documented, it is a vector of zeros
        
$$
\mathbf{f}_i^\mathrm{C} = \sum_{j=1}^{N_i} w_j \ \mathrm{cvec}_j
$$
        
        - There are different options for calculating the weights (`no_structure_policy` and `no_chemosensory_policy`, each of which can be either 'include', 'zero', or 'remove') but they are essentially relative amounts of the compounds in the food
        
        - `no_structure_policy` set to:
        
            - 'zero':
            
                - Amounts of compounds without documented structures do not count in weight numerators, but do count in the weight denominator
                
$$
w_j = \frac{a_j \delta_j^\mathrm{struct}}{a_1 + \cdot \cdot \cdot + a_{N_i}}
$$
                
            - 'remove':
            
                - Amounts of compounds without documented structures do not count in either weight numerators or the denominator
                
$$
w_j = \frac{a_j \delta_j^\mathrm{struct}}{a_1 \delta_1^\mathrm{struct} + \cdot\cdot\cdot + a_{N_i} \delta_{N_i}^\mathrm{struct}}
$$
                
        - `no_chemosensory_policy` set to:
        
            - 'include':
            
                - Amounts of compounds without documented chemosensory properties count in weight numerators and the denominator
                
$$
w_j = \frac{a_j}{a_1 + \cdot\cdot\cdot + a_{N_i}}
$$
                
            - 'zero':
            
                - Amounts of compounds without documented chemosensory properties do not count in weight numerators but do count in the denominator
                
$$
w_j = \frac{a_j \delta_j^\mathrm{chemo}}{a_1 + \cdot\cdot\cdot + a_{N_i}}
$$
                
            - 'remove':
            
                - Amounts of compounds without documented chemosensory properties do not count in either weight numerators or the denominator
                
$$
w_j = \frac{a_j \delta_j^\mathrm{chemo}}{a_1 \delta_1^\mathrm{chemo} + \cdot\cdot\cdot + a_{N_i} \delta_{N_i}^\mathrm{chemo}}
$$
                
    - All foods under consideration are assembled into a data matrix
    
$$
\mathbf{X}^\mathrm{C} = [ \mathbf{f}_1^\mathrm{C}, ..., \mathbf{f}_i^\mathrm{C}, ..., \mathbf{f}_N^\mathrm{C} ] ^ \mathrm{T}
$$
    
    - There is an option to z-score this matrix (each feature/column) before doing the pairwise distance or PCA calculations
    
    - Pairwise distance calculations (if you use t-SNE) yield a pairwise distance matrix

$$
\mathbf{D}^\mathrm{C} = [ d(\mathbf{f}_m^\mathrm{C}, \mathbf{f}_n^\mathrm{C}) ] = [ d(\mathbf{X}^\mathrm{C}[m, :], \mathbf{X}^\mathrm{C}[n, :]) ]
$$

    and 2- or 3-dimensional embeddings for each food $\tilde{\mathbf{f}}_i^\mathrm{C}$ are constructed such that their distance structure approximates $\mathbf{D}^\mathrm{C}$

    - Covariance calculations (if you use PCA) yield a covariance matrix
    
$$
\mathbf{C}^\mathrm{C} = [ \mathrm{cov}(\mathbf{X}^\mathrm{C}[:, m], \mathbf{X}^\mathrm{C}[:, n]) ]
$$

    and 2- or 3-dimensional embeddings for each food $\tilde{\mathbf{f}}_i^\mathrm{C}$ are constructed by projecting onto the principal components of the covariance matrix

**Visualization in taste space**

- In the database, each food, $\mathrm{food}_i$, is composed of a list of $N_i$ compounds along with their amounts, whether or not they have a documented chemical structure, and whether or not they have a documented chemosensory quality (taste _or_ aroma):

$$
\mathrm{food}_i \leftrightarrow [ \mathrm{compound}_1, ..., \mathrm{compound}_j, ..., \mathrm{compound}_{N_i} ]
$$

$$
\mathrm{compound}_j \leftrightarrow ( \mathrm{name}_j, a_j, \delta_j^\mathrm{struct}, \delta_j^\mathrm{chemo} )
$$

- If the food has documented chemosensory qualities, they will be listed; the possible qualities are: Bitter, Pungent, Astringent, Sweet, Sour, Cool, Umami, Salty
    
- The taste representation of each food, $\mathbf{f}_i^\mathrm{A}$, is a function of its compounds
    
- Each food is a weighted sum of its compound vectors, each of which is an 8-dimensional binary vector indicating presence or absence of each possible chemosensory quality

$$
\mathbf{f}_i^\mathrm{A} = \sum_{j=1}^{N_i} w_j \ \mathrm{tvec}_j
$$
    
- There are different options for calculating the weights (`no_chemosensory_policy`, which can be either 'zero' or 'remove') but they are essentially relative amounts of the compounds in the food
    
    - `no_chemosensory_policy` set to:
      
        - 'zero':
        
            - Amounts of compounds without documented chemosensory properties do not count in weight numerators but do count in the denominator
            
$$
w_j = \frac{a_j \delta_j^\mathrm{chemo}}{a_1 + \cdot\cdot\cdot + a_{N_i}}
$$
          
        - 'remove':
        
            - Amounts of compounds without documented chemosensory properties do not count in either weight numerators or the denominator
            
$$
w_j = \frac{a_j \delta_j^\mathrm{chemo}}{a_1 \delta_1^\mathrm{chemo} + \cdot\cdot\cdot + a_{N_i} \delta_{N_i}^\mathrm{chemo}}
$$
          
- All foods under consideration are assembled into a data matrix

$$
\mathbf{X}^\mathrm{A} = [ \mathbf{f}_1^\mathrm{A}, ..., \mathbf{f}_i^\mathrm{A}, ..., \mathbf{f}_N^\mathrm{A} ] ^ \mathrm{T}
$$
    
- There is an option to z-score this matrix (each feature/column) before doing the pairwise distance or PCA calculations
    
- Pairwise distance calculations (if you use t-SNE) yield a pairwise distance matrix

$$
\mathbf{D}^\mathrm{A} = [ d(\mathbf{f}_m^\mathrm{A}, \mathbf{f}_n^\mathrm{A}) ] = [ d(\mathbf{X}^\mathrm{A}[m, :], \mathbf{X}^\mathrm{A}[n, :]) ]
$$

and 2- or 3-dimensional embeddings for each food $\tilde{\mathbf{f}}_i^\mathrm{A}$ constructed such that their distance structure approximates $\mathbf{D}^\mathrm{A}$

- Covariance calculations (if you use PCA) yield a covariance matrix

$$
\mathbf{C}^\mathrm{A} = [ \mathrm{cov}(\mathbf{X}^\mathrm{A}[:, m], \mathbf{X}^\mathrm{A}[:, n]) ]
$$

and 2- or 3-dimensional embeddings for each food constructed by projecting onto the principal components of the covariance matrix
