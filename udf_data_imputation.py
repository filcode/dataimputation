def set_up():
    
    #     pip reccomandation: if something get wrong check out the compatibilities beetween libraries

    #     Install:
    #     from sklearn.experimental import enable_iterative_imputer
    #     from sklearn.impute import IterativeImputer
    #     from sklearn.linear_model import LinearRegression
    #     from sklearn.linear_model import BayesianRidge
    #     from sklearn.impute import KNNImputer

    #     # Miceforest (MICE)
    #     import miceforest as mf
    #     from miceforest import ImputationKernel
    #     from sklearn.metrics import mean_squared_error

    #     # General using
    #     from sklearn.metrics import mean_squared_error
    #     from sklearn.model_selection import train_test_split
    return


def sum_up_count_values_df(df):
    '''
    Sum up:
    - dataframe size
    - count total values not nan
    - count total values nan
    '''
    print(f'Total df values (non-numeric and numeric): {df.size}')
    print(f'Total values (numeric): {df.select_dtypes(include=[np.number]).count().sum().sum()}')
    print(f'Total values (numeric): {df.select_dtypes(include=[np.number]).isna().sum().sum()}')
    return

def iterative_imputation_knn(df, test_column = 'HICP(%)'):
    col_to_test = test_column
    true_values_list = []
    imputed_values_list = []
    
    '''
    KNN used
    Made by each subgroup by 'geo' column
    '''

    groups = df.groupby('geo', group_keys=False)


    def impute_group(group):
        train, test = train_test_split(group.dropna(subset=[col_to_test]), test_size=0.2, random_state=42)
        test_copy = test.copy()
        missing_rows = test_copy.sample(frac=0.1).index
        true_values = test_copy.loc[missing_rows, col_to_test]
        test_copy.loc[missing_rows, col_to_test] = np.nan

        numerical_columns = train.select_dtypes(include=[np.number]).columns
        imp = KNNImputer(n_neighbors=2, weights='uniform')
        imp.fit(train[numerical_columns])
        imputed_test = imp.transform(test_copy[numerical_columns])
        imputed_test_df = pd.DataFrame(imputed_test, columns=numerical_columns, index=test.index)

        true_values_list.extend(true_values)
        imputed_values_list.extend(imputed_test_df.loc[missing_rows, col_to_test])

        imputed_group = pd.concat([train, imputed_test_df], sort=False)

        return imputed_group

    imputed_df = groups.apply(impute_group).reset_index(drop=True)

    comparison_df = pd.DataFrame({
        'True Values': true_values_list,
        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values_list]
    })

    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))

    range_of_data = df[test_column].max() - df[test_column].min()
    percentage_error = (rmse / range_of_data) * 100
    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')
    
    display(comparison_df)

    return imputed_df, comparison_df, rmse


def iterative_imputation_sklearn(df, estimator_choice='BayesianRidge', col_to_test = 'HICP(%)', geo_col = 'geo'):
    
    '''
    Impute missing values using iterative imputer.
    
    Parameters:
    - df: DataFrame
    - estimator_choice: Either 'BayesianRidge' or 'LinearRegression'
    
    Returns:
    imputed_df, comparison_df, rmse
    '''
     # Check estimator choice
    if estimator_choice not in ['BayesianRidge', 'LinearRegression']:
        raise ValueError("estimator_choice must be either 'BayesianRidge' or 'LinearRegression'")

    # Select column to check
    
    true_values_list = []
    imputed_values_list = []
    categorical_values_list = []

    # Group by 'geo' column
    groups = df.groupby(geo_col)
    
    # Model used
    print('BayesianRidge model')

    # Define a function to impute missing values within each group
    def impute_group(group):
        subset_group = group.dropna(subset=[col_to_test]).copy()
        true_values = subset_group.sample(frac=0.1)[col_to_test]
        subset_group.loc[true_values.index, col_to_test] = np.nan
        numerical_columns = subset_group.select_dtypes(include=[np.number]).columns
        
        if estimator_choice == 'BayesianRidge':
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=30, random_state=0)
        else:
            imp = IterativeImputer(estimator=LinearRegression(), max_iter=30, random_state=0)
            
        subset_group[numerical_columns] = imp.fit_transform(subset_group[numerical_columns])
        
        categorical_columns = subset_group.select_dtypes(exclude=[np.number]).columns

        #         # --- 3-KNN
#         imp = KNNImputer(n_neighbors=2, weights='uniform')
#         subset_group[numerical_columns] = imp.fit_transform(subset_group[numerical_columns])
        
        for idx in true_values.index:
            true_values_list.append(true_values[idx])
            imputed_values_list.append(subset_group.loc[idx, col_to_test])
            categorical_values_list.append(group.loc[idx, categorical_columns])

        return subset_group

    imputed_df = groups.apply(impute_group).reset_index(drop=True)

    comparison_df = pd.DataFrame({
        **pd.DataFrame(categorical_values_list),
        'True Values': true_values_list,
        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values_list]
    })

    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))
    range_of_data = df[col_to_test].max() - df[col_to_test].min()
    percentage_error = (rmse / range_of_data) * 100
    
    print(f'MEAN true values: {np.mean(true_values_list):.2f}')
    print(f'MEAN imputed values: {np.mean(imputed_values_list):.2f}')
    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')
    
    display(comparison_df)

    return imputed_df, comparison_df, rmse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def mice_train_test(X, y):
    
    '''
    # Make sure you columns 'geo', 'country' and 'year' are category
    
    Steps model:
    1) Imputation MICE for X train (saved in 'X_train_imputed')
    2) Train LR model in 'X_train_imputed'
    3) Imputation MICE for X test (saved in 'X_test_imputed') 
    4) predictions variable: model predict 'X_test_imputed'
    Results beetwen y_test, predictions: rmse, r2, adjusted_r2
    
    ====================
   
    - Select original dataframe
    - Split in: X, y
    - test_size setted: 0.2 (suggestion to use train test: with enough quantity of data)
    
    Note: make sure you have enough nan value to make MICE interesting to use
    '''

    # Dividiamo i dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Inizializziamo il kernel di miceforest per il training set
    kernel_train = mf.ImputationKernel(X_train, save_all_iterations=True, random_state=0)

    # Imputiamo i valori mancanti nel training set
    kernel_train.mice(3)

    # Recuperiamo il dataset imputato
    X_train_imputed = kernel_train.complete_data()
    X_train_df = pd.DataFrame(X_train_imputed)

    # Addestriamo un modello sui dati imputati
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    # Imputiamo i valori mancanti nel test set usando le stesse informazioni dal training set
    kernel_test = mf.ImputationKernel(X_test, save_all_iterations=True, random_state=0)
    kernel_test.mice(3)

    # Recuperiamo il dataset imputato
    X_test_imputed = kernel_test.complete_data()
    X_test_df = pd.DataFrame(X_test_imputed)

    # Root Mean Squared Error: calcoliamo l'errore quadratico medio del modello sul test set
    predictions = model.predict(X_test_imputed)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'Root Mean Squared Error: {rmse}')

    # R2
    r2 = r2_score(y_test, predictions)
    print(f'R-squared: {r2}')

    # R2 Adjusted
    n = y_test.shape[0] # il numero di osservazioni
    p = X_test.shape[1] # il numero di predittori

    r2 = r2_score(y_test, predictions)
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    print(f'Adjusted R-squared: {adjusted_r2}')
    
    return rmse, r2, adjusted_r2

# test one: made by all data
def mice_imputation_test(df, col_to_test = 'HICP(%)', cols_to_category=['geo', 'country', 'year']):
    '''
    **Test Miceforest predictions**
    
    Select original dataframe (with nan values)
    
    1) Select df with original values
    1) Select columns to test. That values will be randomly converted to nan and tested with predictions
    2) All nan values predicted will be compared with real
    
    Note: Make sure you don't have empty values in columns: 'geo', 'country', 'year' 
    '''
    
    # If columns are not category yet, convert them
    df_mf = df.copy()
    df_mf[cols_to_category] = df_mf[cols_to_category].astype('category')
    
    # Crea una copia del DataFrame e seleziona una frazione dei dati noti da imputare
    df_mf = df.dropna(subset=[col_to_test]).copy()
    true_values = df_mf.sample(frac=0.1)[col_to_test]
    df_mf.loc[true_values.index, col_to_test] = np.nan
    
    # --- MICE model creating
    kds = mf.ImputationKernel(df_mf, random_state=42)
    
    # --- Tuning parameters
    optimal_parameters, losses = kds.tune_parameters(dataset=0, optimization_steps=10)
    kds.mice(1, variable_parameters=optimal_parameters)
    # print(optimal_parameters)
    # ---
    
    # --- Manual setting
    # n_estimators, are number of tree. So, define it considering how mauch data your df contains
    # kds.mice(iterations=20, n_estimators=10, device='gpu') # verbose=2


    # --- Completa i dati imputati
    imputed_df = kds.complete_data()

    # Recupera i valori imputati
    imputed_values = imputed_df.loc[true_values.index, col_to_test]

    # --- Calcola l'RMSE
    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))

    # Calcola l'errore percentuale
    range_of_data = df[col_to_test].max() - df[col_to_test].min()
    percentage_error = (rmse / range_of_data) * 100
    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')

    # --- Crea un DataFrame con i valori reali e imputati
    comparison_df = pd.DataFrame({
        'True Values': true_values,
        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values]
    })
    
    display(comparison_df)

    return imputed_df, comparison_df, rmse, kds


def mice_forest_geo(df, col_to_test='HICP(%)', geo_column='geo', country_column='country'):
    
    '''
    This function allows to impute the missing data in a dataframe taking into account
    the subgroups formed by the "geo" column, providing an estimate of the accuracy of
    the imputation and returning the imputed data.
    
    Decision has been made because in this way prediction will take into cosnderation countries similarities.
    
    
    Results:
    - All imputed dataframes are combined into a single dataframe.
    - The Root Mean Squared Error (RMSE) is calculated between the real and imputed values.
    - A dataframe is created to compare the real and imputed values.    
    ============
    
    Required:
    - df: original dataframe
    - col_to_test: column to use fo testing prediction
    - geo_column: geo column in dataframe
    - country_column: country column in dataframe
    
    ============
    
    NOTE: mice models used consider also outliers values, because our goal si to identify values that
    could be consistent with period and country reference. Decision to consider outliers has been done
    because outliers are not wrong values but correct one and second because if we remove potentially
    value predicted in same outlier row will be not realistic.
    '''
    
    df_geo_list = df[geo_column].unique()
    
    # If you have enough data, can also subgrouping by 'year', so it will give as reference the same period (probably more attendible)

    # Lista per contenere i dataframe imputati
    imputed_dfs = []
    true_values_list = []
    imputed_values_list = []

    for geo_value in df_geo_list:
        # Estrai il sottogruppo corrispondente al valore corrente di 'geo'
        group = df[df[geo_column] == geo_value].copy()

        # Seleziona una frazione dei dati noti da imputare
        true_values = group.sample(frac=0.1)[col_to_test]
        group.loc[true_values.index, col_to_test] = np.nan

        # Rimuovi le categorie inutilizzate nelle colonne "geo" e "country"
        group[geo_column] = group[geo_column].cat.remove_unused_categories()
        group[country_column] = group[country_column].cat.remove_unused_categories()
        
        # --- MICE
        
        # Crea il kernel di imputazione
        kds = ImputationKernel(group, random_state=4)
        
        # Manually application
        kds.mice(iterations=5, n_estimators=50, device='gpu') # verbose=2
        # ---
        
        # Tuning parameters
	# optimal_parameters, losses = kds.tune_parameters(dataset=0, optimization_steps=10)
        # kds.mice(1, variable_parameters=optimal_parameters)
        # print(optimal_parameters)
        # ---
        
        # Ottieni il dataframe imputato
        df_imputed = kds.complete_data()
        # ---
        
        # Ripristina le categorie originali se necessario
        df_imputed[geo_column] = pd.Categorical(df_imputed[geo_column], categories=df[geo_column].unique())
        df_imputed[country_column] = pd.Categorical(df_imputed[country_column], categories=df[country_column].unique())

        # Aggiungi il dataframe imputato alla lista
        imputed_dfs.append(df_imputed)

        # Recupera i valori imputati
        imputed_values = df_imputed.loc[true_values.index, col_to_test]

        # Estendi le liste dei valori veri e imputati
        true_values_list.extend(true_values)
        imputed_values_list.extend(imputed_values)

    # Combina i dataframe imputati in un unico dataframe
    final_df = pd.concat(imputed_dfs)

    # Calcola l'RMSE
    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))
    print(f'RMSE: {rmse:.2f}')

    # Crea un DataFrame con i valori reali e imputati
    comparison_df = pd.DataFrame({
        'True Values': true_values_list,
        'Imputed Values': imputed_values_list
    })

    # Visualizza il DataFrame di confronto
    display(comparison_df)
    
    return final_df, rmse, comparison_df

