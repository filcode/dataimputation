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

def iterative_imputation_knn(df, test_column = 'HICP(%)', grouping_col = 'geo', size_test=0.2):
    
    # Suggestion to use train test: with enough quantity of data
    
    '''
    KNN made by each distinct values of 'geo' column
    '''
    
    col_to_test = test_column
    
    true_values_list = []
    imputed_values_list = []
    
    groups = df.groupby(grouping_col, group_keys=False)

    def impute_group(group):
        train, test = train_test_split(group.dropna(subset=[col_to_test]), test_size=size_test, random_state=42)
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

def iterative_imputation_sklearn(df, estimator_choice='BayesianRidge', col_to_test = 'HICP(%)', grouping_col = 'geo'):
    
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
    groups = df.groupby(grouping_col)
    
    # Model used
    print('BayesianRidge model')

    # Define a function to impute missing values within each group
    def impute_group(group):
        subset_group = group.dropna(subset=[col_to_test]).copy()
        true_values = subset_group.sample(frac=0.1)[col_to_test]
        subset_group.loc[true_values.index, col_to_test] = np.nan
        numerical_columns = subset_group.select_dtypes(include=[np.number]).columns
        
        # -- model decided to use in udf setting
        if estimator_choice == 'BayesianRidge':
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=30, random_state=0)
        else:
            imp = IterativeImputer(estimator=LinearRegression(), max_iter=30, random_state=0)
            
        subset_group[numerical_columns] = imp.fit_transform(subset_group[numerical_columns])
        
        categorical_columns = subset_group.select_dtypes(exclude=[np.number]).columns
        # --
        
        for idx in true_values.index:
            true_values_list.append(true_values[idx])
            imputed_values_list.append(subset_group.loc[idx, col_to_test])
            categorical_values_list.append(group.loc[idx, categorical_columns])

        return subset_group

    imputed_df = groups.apply(impute_group).reset_index(drop=True)
    
    # -- df comparing values
    comparison_df = pd.DataFrame({
        **pd.DataFrame(categorical_values_list),
        'True Values': true_values_list,
        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values_list]
    })
    
    # -- Evaluate model
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
    # Make sure your 'object' columns are converted to 'category' type
    (category are more dreadable for ml models)
    
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

    # Train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Kernel initialization for miceforest training
    kernel_train = mf.ImputationKernel(X_train, save_all_iterations=True, random_state=0)

    # impute the missing values in the training set
    kernel_train.mice(3)

    # recover the imputed dataset
    X_train_imputed = kernel_train.complete_data()
    X_train_df = pd.DataFrame(X_train_imputed)

    # Train imputed data
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    # We input the missing values in the test set using the same information from the training set
    kernel_test = mf.ImputationKernel(X_test, save_all_iterations=True, random_state=0)
    kernel_test.mice(3)

    # recover the imputed dataset
    X_test_imputed = kernel_test.complete_data()
    X_test_df = pd.DataFrame(X_test_imputed)

    # Root Mean Squared Error:  calculate the Model Standard Quadratic Error on the test set
    predictions = model.predict(X_test_imputed)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'Root Mean Squared Error: {rmse}')

    # R2
    r2 = r2_score(y_test, predictions)
    print(f'R-squared: {r2}')

    # R2 Adjusted
    n = y_test.shape[0] # the number of observations
    p = X_test.shape[1] # the number of predictors

    r2 = r2_score(y_test, predictions)
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    print(f'Adjusted R-squared: {adjusted_r2}')
    
    return rmse, r2, adjusted_r2


# test one: made by all data
def mice_imputation_test(df, col_to_test = 'HICP(%)', cols_to_category=['geo', 'country', 'year']):
    
    
    '''
    --> test one: made using all data for imputation
     
    =================
    
    Impute need:
    
    Select original dataframe (with nan values)
    
    1) Select df with original values
    1) Select columns to test. That values will be randomly converted to nan and tested with predictions
    2) All nan values predicted will be compared with real
    
    Note: Make sure you don't have empty values in columns: 'geo', 'country', 'year'
    
    =================
    
    Output:
    
    - kds: imputed data
    - imputed_df: dataframe imputed
    - comparison_df: dataframe with true values and imputed values
    - rmse: Root Mean Squared Error (RMSE) is calculated between the real and imputed values.
    
    '''
    
    # If columns are not category yet, convert them
    df_mf = df.copy()
    df_mf[cols_to_category] = df_mf[cols_to_category].astype('category')
    
    # Create a copy of the DataFrame and select a fraction of the known data to be imputed
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


    # --- Complete the imputed data
    imputed_df = kds.complete_data()

    # Recovers the imputed values
    imputed_values = imputed_df.loc[true_values.index, col_to_test]

    # --- Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))

    # Calculate the percentage error
    range_of_data = df[col_to_test].max() - df[col_to_test].min()
    percentage_error = (rmse / range_of_data) * 100
    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')

    # --- Create a DataFrame with real and imputed values
    comparison_df = pd.DataFrame({
        'True Values': true_values,
        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values]
    })
    
    display(comparison_df)

    return imputed_df, comparison_df, rmse, kds

# test two: made by each 'geo'
def mice_forest_grouped(df, col_to_test='HICP(%)', grouping_column='geo', other_cat_col='country'): # other_cat_column (no year)
    
    '''
    --> test two: imputation made by each 'geo' distinct value
     
    =================
    
    This function allows to impute the missing data in a dataframe taking into account
    the subgroups formed by the "geo" column, providing an estimate of the accuracy of
    the imputation and returning the imputed data.
    
    Decision has been made because in this way prediction will take into cosnderation countries similarities.
      
    ============
    
    Imput needs:
    - df: original dataframe
    - col_to_test: column to use fo testing prediction
    - geo_column: geo column in dataframe
    - country_column: country column in dataframe
    
    ============
    
    Results:
    - final_df: all imputed dataframes are combined into a single dataframe.
    - rmse: Root Mean Squared Error (RMSE) is calculated between the real and imputed values.
    - comparison_df: a dataframe is created to compare the real and imputed values.

    '''
    
    df_geo_list = df[geo_column].unique()
    
    # If you have enough data, can also subgrouping by 'year', so it will give as reference the same period (probably more attendible)

    # List to contain imputed dataframes
    imputed_dfs = []
    true_values_list = []
    imputed_values_list = []

    for geo_value in df_geo_list:
        # Extract the subgroup corresponding to the current value of 'geo'
        group = df[df[geo_column] == geo_value].copy()

        # Select a fraction of known data to be imputed
        true_values = group.sample(frac=0.1)[col_to_test]
        group.loc[true_values.index, col_to_test] = np.nan

        # Remove unused categories in "geo" and "country" columns
        group[geo_column] = group[geo_column].cat.remove_unused_categories()
        group[other_cat_col] = group[other_cat_col].cat.remove_unused_categories()
        
        # --- MICE
        
        # Create the attribution kernel
        kds = ImputationKernel(group, random_state=4)
        
        # Manually application
        kds.mice(iterations=5, n_estimators=50, device='gpu') # verbose=2
        # ---
        
        # Tuning parameters
        # optimal_parameters, losses = kds.tune_parameters(dataset=0, optimization_steps=10)
        # kds.mice(1, variable_parameters=optimal_parameters)
        # print(optimal_parameters)
        # ---
        
        # Get the imputed dataframe
        df_imputed = kds.complete_data()
        # ---
        
        # Restore original categories if necessary
        df_imputed[geo_column] = pd.Categorical(df_imputed[geo_column], categories=df[geo_column].unique())
        df_imputed[other_cat_col] = pd.Categorical(df_imputed[other_cat_col], categories=df[other_cat_col].unique())

        # Add the imputed dataframe to the list
        imputed_dfs.append(df_imputed)

        # Recovers the imputed values
        imputed_values = df_imputed.loc[true_values.index, col_to_test]

        # Extend real and imputed value lists
        true_values_list.extend(true_values)
        imputed_values_list.extend(imputed_values)

    # Combine imputed dataframes into a single dataframe
    final_df = pd.concat(imputed_dfs)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))
    print(f'RMSE: {rmse:.2f}')

    # Create a DataFrame with real and imputed values
    comparison_df = pd.DataFrame({
        'True Values': true_values_list,
        'Imputed Values': imputed_values_list
    })

    # Visualizza il DataFrame di confronto
    display(comparison_df)
    
    return final_df, rmse, comparison_df


