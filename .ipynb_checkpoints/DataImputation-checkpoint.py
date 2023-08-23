{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6eefac37",
   "metadata": {},
   "source": [
    "### Data imputation\n",
    "You have to option in order to replace NaN values left:\n",
    "1. Get real values from official website\n",
    "2. Imputation (considering that value can not be aligned with reality completely)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b77b6ba",
   "metadata": {},
   "source": [
    "**SetUp**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "09204303",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install miceforest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "88401464",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install pyarrow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "42cf6cdf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install --upgrade pandas pyarrow miceforest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d1a956c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# library installation (check it out to see how it works)\n",
    "# !pip install git+https://github.com/AnotherSamWilson/miceforest.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "10aad7f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install miceforest --no-cache-dir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c08f620",
   "metadata": {},
   "outputs": [],
   "source": [
    "# library installation (check it out to see how it works)\n",
    "# !pip install git+https://github.com/AnotherSamWilson/miceforest.git\n",
    "# !pip install miceforest --no-cache-dir"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "168a51a2",
   "metadata": {},
   "source": [
    "<!--  -->"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b8e973ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.experimental import enable_iterative_imputer\n",
    "from sklearn.impute import IterativeImputer\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import BayesianRidge\n",
    "from sklearn.impute import KNNImputer\n",
    "\n",
    "# Miceforest (MICE)\n",
    "import miceforest as mf\n",
    "from miceforest import ImputationKernel\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# General using\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed34e8b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def set_up():\n",
    "    '''\n",
    "    pip reccomandation: if something get wrong check out the compatibilities beetween libraries\n",
    "    \n",
    "    Install:\n",
    "    from sklearn.experimental import enable_iterative_imputer\n",
    "    from sklearn.impute import IterativeImputer\n",
    "    from sklearn.linear_model import LinearRegression\n",
    "    from sklearn.linear_model import BayesianRidge\n",
    "    from sklearn.impute import KNNImputer\n",
    "\n",
    "    # Miceforest (MICE)\n",
    "    import miceforest as mf\n",
    "    from miceforest import ImputationKernel\n",
    "    from sklearn.metrics import mean_squared_error\n",
    "\n",
    "    # General using\n",
    "    from sklearn.metrics import mean_squared_error\n",
    "    from sklearn.model_selection import train_test_split\n",
    "    '''\n",
    "    return"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbbd7fd3",
   "metadata": {},
   "source": [
    "<!--  -->"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f6c27c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sum_up_count_values_df(df):\n",
    "    '''\n",
    "    Sum up:\n",
    "    - dataframe size\n",
    "    - count total values not nan\n",
    "    - count total values nan\n",
    "    '''\n",
    "    print(f'Total df values (non-numeric and numeric): {df.size}')\n",
    "    print(f'Total values (numeric): {df.select_dtypes(include=[np.number]).count().sum().sum()}')\n",
    "    print(f'Total values (numeric): {df.select_dtypes(include=[np.number]).isna().sum().sum()}')\n",
    "    return"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4370ad9",
   "metadata": {},
   "source": [
    "<!--  -->"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "deeb1c37",
   "metadata": {},
   "source": [
    "<u>*!!! Suggestion to use train test: with enough quantity of data*</u>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "681852b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def iterative_imputation_knn(df, test_column = 'HICP(%)'):\n",
    "    col_to_test = test_column\n",
    "    true_values_list = []\n",
    "    imputed_values_list = []\n",
    "    \n",
    "    '''\n",
    "    KNN used\n",
    "    Made by each subgroup by 'geo' column\n",
    "    '''\n",
    "\n",
    "    groups = df.groupby('geo', group_keys=False)\n",
    "\n",
    "    def impute_group(group):\n",
    "        train, test = train_test_split(group.dropna(subset=[col_to_test]), test_size=0.2, random_state=42)\n",
    "        test_copy = test.copy()\n",
    "        missing_rows = test_copy.sample(frac=0.1).index\n",
    "        true_values = test_copy.loc[missing_rows, col_to_test]\n",
    "        test_copy.loc[missing_rows, col_to_test] = np.nan\n",
    "\n",
    "        numerical_columns = train.select_dtypes(include=[np.number]).columns\n",
    "        imp = KNNImputer(n_neighbors=2, weights='uniform')\n",
    "        imp.fit(train[numerical_columns])\n",
    "        imputed_test = imp.transform(test_copy[numerical_columns])\n",
    "        imputed_test_df = pd.DataFrame(imputed_test, columns=numerical_columns, index=test.index)\n",
    "\n",
    "        true_values_list.extend(true_values)\n",
    "        imputed_values_list.extend(imputed_test_df.loc[missing_rows, col_to_test])\n",
    "\n",
    "        imputed_group = pd.concat([train, imputed_test_df], sort=False)\n",
    "\n",
    "        return imputed_group\n",
    "\n",
    "    imputed_df = groups.apply(impute_group).reset_index(drop=True)\n",
    "\n",
    "    comparison_df = pd.DataFrame({\n",
    "        'True Values': true_values_list,\n",
    "        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values_list]\n",
    "    })\n",
    "\n",
    "    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))\n",
    "\n",
    "    range_of_data = df[test_column].max() - df[test_column].min()\n",
    "    percentage_error = (rmse / range_of_data) * 100\n",
    "    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')\n",
    "    \n",
    "    display(comparison_df)\n",
    "\n",
    "    return imputed_df, comparison_df, rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e14c8e4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "imputed_df, comparison_df, rmse = iterative_imputation_knn(df_test_rep, test_column = 'HICP(%)'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5628eae6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def iterative_imputation_sklearn(df, estimator_choice='BayesianRidge', col_to_test = 'HICP(%)', geo_col = 'geo'):\n",
    "    \n",
    "    '''\n",
    "    Impute missing values using iterative imputer.\n",
    "    \n",
    "    Parameters:\n",
    "    - df: DataFrame\n",
    "    - estimator_choice: Either 'BayesianRidge' or 'LinearRegression'\n",
    "    \n",
    "    Returns:\n",
    "    imputed_df, comparison_df, rmse\n",
    "    '''\n",
    "     # Check estimator choice\n",
    "    if estimator_choice not in ['BayesianRidge', 'LinearRegression']:\n",
    "        raise ValueError(\"estimator_choice must be either 'BayesianRidge' or 'LinearRegression'\")\n",
    "\n",
    "    # Select column to check\n",
    "    \n",
    "    true_values_list = []\n",
    "    imputed_values_list = []\n",
    "    categorical_values_list = []\n",
    "\n",
    "    # Group by 'geo' column\n",
    "    groups = df.groupby(geo_col)\n",
    "    \n",
    "    # Model used\n",
    "    print('BayesianRidge model')\n",
    "\n",
    "    # Define a function to impute missing values within each group\n",
    "    def impute_group(group):\n",
    "        subset_group = group.dropna(subset=[col_to_test]).copy()\n",
    "        true_values = subset_group.sample(frac=0.1)[col_to_test]\n",
    "        subset_group.loc[true_values.index, col_to_test] = np.nan\n",
    "        numerical_columns = subset_group.select_dtypes(include=[np.number]).columns\n",
    "        \n",
    "        if estimator_choice == 'BayesianRidge':\n",
    "            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=30, random_state=0)\n",
    "        else:\n",
    "            imp = IterativeImputer(estimator=LinearRegression(), max_iter=30, random_state=0)\n",
    "            \n",
    "        subset_group[numerical_columns] = imp.fit_transform(subset_group[numerical_columns])\n",
    "        \n",
    "        categorical_columns = subset_group.select_dtypes(exclude=[np.number]).columns\n",
    "\n",
    "        #         # --- 3-KNN\n",
    "#         imp = KNNImputer(n_neighbors=2, weights='uniform')\n",
    "#         subset_group[numerical_columns] = imp.fit_transform(subset_group[numerical_columns])\n",
    "        \n",
    "        for idx in true_values.index:\n",
    "            true_values_list.append(true_values[idx])\n",
    "            imputed_values_list.append(subset_group.loc[idx, col_to_test])\n",
    "            categorical_values_list.append(group.loc[idx, categorical_columns])\n",
    "\n",
    "        return subset_group\n",
    "\n",
    "    imputed_df = groups.apply(impute_group).reset_index(drop=True)\n",
    "\n",
    "    comparison_df = pd.DataFrame({\n",
    "        **pd.DataFrame(categorical_values_list),\n",
    "        'True Values': true_values_list,\n",
    "        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values_list]\n",
    "    })\n",
    "\n",
    "    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))\n",
    "    range_of_data = df[col_to_test].max() - df[col_to_test].min()\n",
    "    percentage_error = (rmse / range_of_data) * 100\n",
    "    \n",
    "    print(f'MEAN true values: {np.mean(true_values_list):.2f}')\n",
    "    print(f'MEAN imputed values: {np.mean(imputed_values_list):.2f}')\n",
    "    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')\n",
    "    \n",
    "    display(comparison_df)\n",
    "\n",
    "    return imputed_df, comparison_df, rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef828abd",
   "metadata": {},
   "outputs": [],
   "source": [
    "imputed_df, comparison_df, rmse = iterative_imputation_sklearn(df, estimator_choice='BayesianRidge', col_to_test = 'HICP(%)', geo_col = 'geo')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43402c84",
   "metadata": {},
   "source": [
    "<!--  -->"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e03fd8f8",
   "metadata": {},
   "source": [
    "**<u>MICE model, following tested two approach</u>**\n",
    "* All dataset\n",
    "* Subgrouped by 'geo', for two reason:\n",
    "    * we can not removed bias from years just selecting one year caused by low data\n",
    "    * identifying more comparable countries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7be74936",
   "metadata": {},
   "outputs": [],
   "source": [
    "# df, X, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6967c1bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "\n",
    "def mice_train_test(X, y):\n",
    "    \n",
    "    '''\n",
    "    # Make sure you columns 'geo', 'country' and 'year' are category\n",
    "    \n",
    "    Steps model:\n",
    "    1) Imputation MICE for X train (saved in 'X_train_imputed')\n",
    "    2) Train LR model in 'X_train_imputed'\n",
    "    3) Imputation MICE for X test (saved in 'X_test_imputed') \n",
    "    4) predictions variable: model predict 'X_test_imputed'\n",
    "    Results beetwen y_test, predictions: rmse, r2, adjusted_r2\n",
    "    \n",
    "    ====================\n",
    "   \n",
    "    - Select original dataframe\n",
    "    - Split in: X, y\n",
    "    - test_size setted: 0.2 (suggestion to use train test: with enough quantity of data)\n",
    "    \n",
    "    Note: make sure you have enough nan value to make MICE interesting to use\n",
    "    '''\n",
    "\n",
    "    # Dividiamo i dati in training e test set\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\n",
    "\n",
    "    # Inizializziamo il kernel di miceforest per il training set\n",
    "    kernel_train = mf.ImputationKernel(X_train, save_all_iterations=True, random_state=0)\n",
    "\n",
    "    # Imputiamo i valori mancanti nel training set\n",
    "    kernel_train.mice(3)\n",
    "\n",
    "    # Recuperiamo il dataset imputato\n",
    "    X_train_imputed = kernel_train.complete_data()\n",
    "    X_train_df = pd.DataFrame(X_train_imputed)\n",
    "\n",
    "    # Addestriamo un modello sui dati imputati\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train_imputed, y_train)\n",
    "\n",
    "    # Imputiamo i valori mancanti nel test set usando le stesse informazioni dal training set\n",
    "    kernel_test = mf.ImputationKernel(X_test, save_all_iterations=True, random_state=0)\n",
    "    kernel_test.mice(3)\n",
    "\n",
    "    # Recuperiamo il dataset imputato\n",
    "    X_test_imputed = kernel_test.complete_data()\n",
    "    X_test_df = pd.DataFrame(X_test_imputed)\n",
    "\n",
    "    # Root Mean Squared Error: calcoliamo l'errore quadratico medio del modello sul test set\n",
    "    predictions = model.predict(X_test_imputed)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, predictions))\n",
    "    print(f'Root Mean Squared Error: {rmse}')\n",
    "\n",
    "    # R2\n",
    "    r2 = r2_score(y_test, predictions)\n",
    "    print(f'R-squared: {r2}')\n",
    "\n",
    "    # R2 Adjusted\n",
    "    n = y_test.shape[0] # il numero di osservazioni\n",
    "    p = X_test.shape[1] # il numero di predittori\n",
    "\n",
    "    r2 = r2_score(y_test, predictions)\n",
    "    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))\n",
    "    print(f'Adjusted R-squared: {adjusted_r2}')\n",
    "    \n",
    "    return rmse, r2, adjusted_r2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8be5b9c4",
   "metadata": {},
   "source": [
    "<!--  -->"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ac2e892",
   "metadata": {},
   "source": [
    "<u>*!!! Suggestion to use train test: with enough quantity of data*</u>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "58491bf9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# test one: made by all data\n",
    "def mice_imputation_test(df, col_to_test = 'HICP(%)', cols_to_category=['geo', 'country', 'year']):\n",
    "    '''\n",
    "    **Test Miceforest predictions**\n",
    "    \n",
    "    Select original dataframe (with nan values)\n",
    "    \n",
    "    1) Select df with original values\n",
    "    1) Select columns to test. That values will be randomly converted to nan and tested with predictions\n",
    "    2) All nan values predicted will be compared with real\n",
    "    \n",
    "    Note: Make sure you don't have empty values in columns: 'geo', 'country', 'year' \n",
    "    '''\n",
    "    \n",
    "    # If columns are not category yet, convert them\n",
    "    df_mf = df.copy()\n",
    "    df_mf[cols_to_category] = df_mf[cols_to_category].astype('category')\n",
    "    \n",
    "    # Crea una copia del DataFrame e seleziona una frazione dei dati noti da imputare\n",
    "    df_mf = df.dropna(subset=[col_to_test]).copy()\n",
    "    true_values = df_mf.sample(frac=0.1)[col_to_test]\n",
    "    df_mf.loc[true_values.index, col_to_test] = np.nan\n",
    "    \n",
    "    # --- MICE model creating\n",
    "    kds = mf.ImputationKernel(df_mf, random_state=42)\n",
    "    \n",
    "    # --- Tuning parameters\n",
    "    optimal_parameters, losses = kds.tune_parameters(dataset=0, optimization_steps=10)\n",
    "    kds.mice(1, variable_parameters=optimal_parameters)\n",
    "    # print(optimal_parameters)\n",
    "    # ---\n",
    "    \n",
    "    # --- Manual setting\n",
    "    # n_estimators, are number of tree. So, define it considering how mauch data your df contains\n",
    "    # kds.mice(iterations=20, n_estimators=10, device='gpu') # verbose=2\n",
    "\n",
    "\n",
    "    # --- Completa i dati imputati\n",
    "    imputed_df = kds.complete_data()\n",
    "\n",
    "    # Recupera i valori imputati\n",
    "    imputed_values = imputed_df.loc[true_values.index, col_to_test]\n",
    "\n",
    "    # --- Calcola l'RMSE\n",
    "    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))\n",
    "\n",
    "    # Calcola l'errore percentuale\n",
    "    range_of_data = df[col_to_test].max() - df[col_to_test].min()\n",
    "    percentage_error = (rmse / range_of_data) * 100\n",
    "    print(f'RMSE percentage_error (min,max): {percentage_error:.2f}')\n",
    "\n",
    "    # --- Crea un DataFrame con i valori reali e imputati\n",
    "    comparison_df = pd.DataFrame({\n",
    "        'True Values': true_values,\n",
    "        'Imputed Values': ['{:.1f}'.format(value) for value in imputed_values]\n",
    "    })\n",
    "    \n",
    "    display(comparison_df)\n",
    "\n",
    "    return imputed_df, comparison_df, rmse, kds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d900795",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Upload dataframe (df_test_rep)\n",
    "imputed_df, comparison_df, rmse, kds = mice_imputation_test(df_test_rep)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d248409b",
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0b6d34e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot compare the distribution by data and data with imputation\n",
    "kds.plot_imputed_distributions(wspace=0.2,hspace=0.4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eadc901a",
   "metadata": {},
   "source": [
    "<u>Model (use 'geo' reference)</u>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d086cc11",
   "metadata": {},
   "outputs": [],
   "source": [
    "# test two: made by each 'geo'\n",
    "def mice_forest_geo(df, col_to_test='HICP(%)', geo_column='geo', country_column='country'):\n",
    "    \n",
    "    '''\n",
    "    This function allows to impute the missing data in a dataframe taking into account\n",
    "    the subgroups formed by the \"geo\" column, providing an estimate of the accuracy of\n",
    "    the imputation and returning the imputed data.\n",
    "    \n",
    "    Decision has been made because in this way prediction will take into cosnderation countries similarities.\n",
    "    \n",
    "    \n",
    "    Results:\n",
    "    - All imputed dataframes are combined into a single dataframe.\n",
    "    - The Root Mean Squared Error (RMSE) is calculated between the real and imputed values.\n",
    "    - A dataframe is created to compare the real and imputed values.    \n",
    "    ============\n",
    "    \n",
    "    Required:\n",
    "    - df: original dataframe\n",
    "    - col_to_test: column to use fo testing prediction\n",
    "    - geo_column: geo column in dataframe\n",
    "    - country_column: country column in dataframe\n",
    "    \n",
    "    ============\n",
    "    \n",
    "    NOTE: mice models used consider also outliers values, because our goal si to identify values that\n",
    "    could be consistent with period and country reference. Decision to consider outliers has been done\n",
    "    because outliers are not wrong values but correct one and second because if we remove potentially\n",
    "    value predicted in same outlier row will be not realistic.\n",
    "    '''\n",
    "    \n",
    "    df_geo_list = df[geo_column].unique()\n",
    "    \n",
    "    # If you have enough data, can also subgrouping by 'year', so it will give as reference the same period (probably more attendible)\n",
    "\n",
    "    # Lista per contenere i dataframe imputati\n",
    "    imputed_dfs = []\n",
    "    true_values_list = []\n",
    "    imputed_values_list = []\n",
    "\n",
    "    for geo_value in df_geo_list:\n",
    "        # Estrai il sottogruppo corrispondente al valore corrente di 'geo'\n",
    "        group = df[df[geo_column] == geo_value].copy()\n",
    "\n",
    "        # Seleziona una frazione dei dati noti da imputare\n",
    "        true_values = group.sample(frac=0.1)[col_to_test]\n",
    "        group.loc[true_values.index, col_to_test] = np.nan\n",
    "\n",
    "        # Rimuovi le categorie inutilizzate nelle colonne \"geo\" e \"country\"\n",
    "        group[geo_column] = group[geo_column].cat.remove_unused_categories()\n",
    "        group[country_column] = group[country_column].cat.remove_unused_categories()\n",
    "        \n",
    "        # --- MICE\n",
    "        \n",
    "        # Crea il kernel di imputazione\n",
    "        kds = ImputationKernel(group, random_state=4)\n",
    "        \n",
    "        # Manually application\n",
    "        kds.mice(iterations=5, n_estimators=50, device='gpu') # verbose=2\n",
    "        # ---\n",
    "        \n",
    "        # Tuning parameters\n",
    "#         optimal_parameters, losses = kds.tune_parameters(dataset=0, optimization_steps=10)\n",
    "#         kds.mice(1, variable_parameters=optimal_parameters)\n",
    "        # print(optimal_parameters)\n",
    "        # ---\n",
    "        \n",
    "        # Ottieni il dataframe imputato\n",
    "        df_imputed = kds.complete_data()\n",
    "        # ---\n",
    "        \n",
    "        # Ripristina le categorie originali se necessario\n",
    "        df_imputed[geo_column] = pd.Categorical(df_imputed[geo_column], categories=df[geo_column].unique())\n",
    "        df_imputed[country_column] = pd.Categorical(df_imputed[country_column], categories=df[country_column].unique())\n",
    "\n",
    "        # Aggiungi il dataframe imputato alla lista\n",
    "        imputed_dfs.append(df_imputed)\n",
    "\n",
    "        # Recupera i valori imputati\n",
    "        imputed_values = df_imputed.loc[true_values.index, col_to_test]\n",
    "\n",
    "        # Estendi le liste dei valori veri e imputati\n",
    "        true_values_list.extend(true_values)\n",
    "        imputed_values_list.extend(imputed_values)\n",
    "\n",
    "    # Combina i dataframe imputati in un unico dataframe\n",
    "    final_df = pd.concat(imputed_dfs)\n",
    "\n",
    "    # Calcola l'RMSE\n",
    "    rmse = np.sqrt(mean_squared_error(true_values_list, imputed_values_list))\n",
    "    print(f'RMSE: {rmse:.2f}')\n",
    "\n",
    "    # Crea un DataFrame con i valori reali e imputati\n",
    "    comparison_df = pd.DataFrame({\n",
    "        'True Values': true_values_list,\n",
    "        'Imputed Values': imputed_values_list\n",
    "    })\n",
    "\n",
    "    # Visualizza il DataFrame di confronto\n",
    "    display(comparison_df)\n",
    "    \n",
    "    return final_df, rmse, comparison_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06885802",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_df, rmse, comparison_df = mice_forest_geo(df_test_rep,col_to_test='HICP(%)', geo_column='geo', country_column='country')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eaffbcb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f0d8c69",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot compare the distribution by data and data with imputation\n",
    "kds.plot_imputed_distributions(wspace=0.2,hspace=0.4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "808bef4e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3f9928a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be75e35b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1cbc11c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aaa55098",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "069ce36d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
