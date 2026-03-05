import os 
import numpy as np                                                                                                                                                                                 
import pandas as pd
import joblib
import itertools

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class engineering():
    """
    Class is used to engineer use features
    Two methods present: (a) brute force method, (b) feature marker

    Note: this should be used before scaling the features and before oversampling.

    args: 
        (1) path_to_file (type:str) - location of the data file with features; last column is taken as the target feature
        (2) path_to_save (type:str) - loacation to save data
        (3) target (type:str) - name of target variable
        (4) features (list) - list of exploratory features
        (5) csv (type:bool) - whether to save as csv

    return: 
        (1) pandas.Dataframe with newly engineered features
    """

    def __init__(self, path_to_file, path_to_save, path_to_test_file, target, features, csv = False):
        self.path_to_save = path_to_save
        self.csv = csv
        
        self.sample_train = joblib.load(path_to_file)
        self.sample_test = joblib.load(path_to_test_file)
        self.all_data_features = pd.DataFrame()
        
        # Define input and target variables
        if isinstance(features, list):
            self.features = features
        else:
            self.features = joblib.load(features) 

        self.target = target

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.features))



    def movecol(self, dataframe, cols_to_move = [], ref_col = '', place = 'after'):
        """
        Function to rearrange columns

        arg: 
            (a) cols_to_move (list) - list of columns to move
            (b) ref_col (type:str) - reference column 
            (c) place (type:str) - whether to move the specified columns 'before' or 'after' the reference column (set to 'after' as default)

        return:
            (a) pandas.Dataframe
        """

        cols = dataframe.columns.tolist()


        if place == 'after':
            s1 = cols[:list(cols).index(ref_col) + 1]
            s2 = cols_to_move


        if place == 'before':
            s1 = cols[:list(cols).index(ref_col)]
            s2 = cols_to_move + [ref_col]
        

        s1 = [i for i in s1 if i not in s2]
        s3 = [i for i in cols if i not in s1 + s2]
        

        return dataframe[s1 + s2 + s3]

    def generate_feature_ratios(self, df, feature_list):
        """
        Generate all ratio features for a given dataframe based on feature_list.
        """
        import numpy as np
    
        new_cols = []
        all_perm = list(itertools.permutations(feature_list, r=2))
        print(f"Total number of permutations: {len(all_perm)}")
    
        for f1, f2 in all_perm:
            col_name = f"{f1}/{f2}"
            try:
                # Ensure columns exist
                if f1 not in df.columns or f2 not in df.columns:
                    print(f"Skipping {col_name}: Missing column {f1} or {f2}")
                    continue
    
                # Safely compute ratios
                computed_series = pd.Series(df[f1].values) / pd.Series(np.where(df[f2] == 0, np.nan, df[f2]))
                
                # Ensure length matches
                if len(computed_series) != len(df):
                    print(f"Skipping {col_name}: Length mismatch")
                    continue
    
                # Assign to DataFrame
                df[col_name] = computed_series
                new_cols.append(col_name)
                
            except Exception as e:
                print(f"Error processing {col_name}: {e}")
                continue
    
        return new_cols, df

    
    def handle_invalid_values(self, df):
        """
        Replace invalid values (e.g., NaN, inf) with 0 for specified columns.
        """
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df
    
    def scale_features(self, train_df, test_df, target_column):
        """
        Scale all features except the target column using MinMaxScaler. Move target column to end.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
    
        # Scale training features
        train_features = train_df.drop(columns=[target_column])
        scaled_train_features = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )
        train_df = pd.concat([scaled_train_features, train_df[target_column]], axis=1)
    
        # Scale testing features
        test_features = test_df.drop(columns=[target_column])
        scaled_test_features = pd.DataFrame(
            scaler.transform(test_features),
            columns=test_features.columns,
            index=test_features.index
        )
        test_df = pd.concat([scaled_test_features, test_df[target_column]], axis=1)
    
        return train_df, test_df

    def brute_force(self, feature_list):
        #     """
        #     Feature engineering using brute force method.
        #     Use features identified to have statistical significance.
            
        #     Args: 
        #         feature_list (list): List of features to use.
            
        #     Returns: 
        #         pandas.DataFrame: Updated self.sample_train and self.sample_test with newly engineered features.
        #         list: List of newly engineered feature column names.
        #     """
        
        # Original columns
        original_train_cols = self.sample_train.columns.tolist()
        original_test_cols = self.sample_test.columns.tolist()
        
        # Generate ratio features
        self.new_cols, df_engineered_train = self.generate_feature_ratios(self.sample_train, feature_list)
        self.new_cols, df_engineered_test = self.generate_feature_ratios(self.sample_test, feature_list)
        
        self.sample_train = df_engineered_train
        self.sample_test = df_engineered_test
        
        # Handle invalid values
        self.sample_train = self.handle_invalid_values(self.sample_train)
        self.sample_test = self.handle_invalid_values(self.sample_test)
        
        # # Scale features
        # self.sample_train, self.sample_test = self.scale_features(self.sample_train, self.sample_test, self.target)
        
        # Track new columns
        latest_train_cols = self.sample_train.columns.tolist()
        self.new_cols = [c for c in latest_train_cols if c not in original_train_cols]
        
        return self.sample_train, self.sample_test, self.new_cols
    
    
    def brute_force_on_all(self, path_to_full_dataset_features, path_to_full_dataset_data, feature_list):
        #     """
        #     Feature engineering using brute force method on all the data.
        #     Use features identified to have statistical significance.
            
        #     Args: 
        #         feature_list (list): List of features to use.
            
        #     Returns: 
        #         pandas.DataFrame: Updated self.sample_train and self.sample_test with newly engineered features.
        #         list: List of newly engineered feature column names.
        #     """
        
        all_data_features_import = joblib.load(path_to_full_dataset_features)
        #print(all_data_features_import)
        
        all_data_import = joblib.load(path_to_full_dataset_data)
        #print(all_data_import)
                
        all_data_features_list_path = path_to_full_dataset_features.replace("s.pkl", "_list.pkl")
        all_data_features_list = joblib.load(all_data_features_list_path)
       
        
        all_data_features_import = all_data_features_import[all_data_features_list]
        print(f"\nTotal Number of Initial Features: {len(all_data_features_list)}")
        #print(all_data_features_import)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
    
        if self.target in all_data_features_import.columns:
            all_data_features = all_data_features_import.drop(columns=self.target)
        else:
            all_data_features = all_data_features_import
        
        print("\nScale Initial Features\n")
        scaled_all_data_features = pd.DataFrame(
            scaler.fit_transform(all_data_features),
            columns=all_data_features.columns,
            index=all_data_features.index
        )
        
        all_data_features = pd.concat([scaled_all_data_features, all_data_import[self.target]], axis=1, ignore_index=False)
        print(all_data_features)
        
        # Original columns
        original_cols = all_data_features.columns.tolist()
        # all_data_features = all_data_features[original_cols]
        
        new_cols = [] 
        all_data_features_engineered = pd.DataFrame()
        
        # Generate ratio features
        new_cols, all_data_features_engineered = self.generate_feature_ratios(all_data_features, feature_list)
        print(all_data_features_engineered)
        
        all_data_features = all_data_features_engineered  # pd.concat([all_data_features, new_cols], axis=1, ignore_index=False)
        
        # Handle invalid values
        all_data_features = self.handle_invalid_values(all_data_features)
        print(all_data_features)
                
        # # Scale features
        # self.sample_train, self.sample_test = self.scale_features(self.sample_train, self.sample_test, self.target)
        
        ref_col = all_data_features.columns[0]
        
        all_data_features['is_experimental'] = [0.0 for _ in range(len(all_data_features))]
        
        all_data_features = self.movecol(all_data_features, cols_to_move=[self.target, 'is_experimental'], ref_col=ref_col , place='before')
        
        print("\nCheck if features for all data are identical to original sample_train")
        identical_train_check = self.sample_train.columns.equals(all_data_features.columns)
        print(identical_train_check)
        print("\nCheck if features for all data are identical to original sample_test")
        identical_test_check = self.sample_test.columns.equals(all_data_features.columns)
        print(identical_test_check)
        
        if identical_train_check and identical_test_check:
            print("\nAll data feature column names are identical!")
        else:
            raise ValueError("\nAll data feature column names are not identical")
                
        # Track new columns
        latest_train_cols = all_data_features.columns.tolist()
        new_cols = [c for c in latest_train_cols if c not in original_cols]
        
        self.all_data_features = all_data_features
               
        joblib.dump(all_data_features,f"{self.path_to_save}/df_all_data_engineered_features_{self.target}.pkl")
        print(f"\nAll data with engineered features saved to {self.path_to_save}/df_all_data_engineered_features_{self.target}.pkl")
        
        return self.all_data_features
        
        # return self.sample_train, self.sample_test, self.new_cols
    

    def feature_markers(self, feature_list):
        """
        Create feature markers (1 or 0) to indicate the presence or absence of a feature, respectively
        Use features identified to have statistical significance 
            
        args: 
            (1) feature_list (list) - list of features to use

        return: 
            (1) pandas.Dataframe with newly engineered features
        """

        # Compute the feature values of the new features
        for f in feature_list:
            self.sample_train[str(f) + '_marker'] = self.sample_train[f].apply(lambda x: 1 if x != 0 else 0)


        # Move target to last column
        self.sample_train = self.movecol(self.sample_train, cols_to_move=[self.target], ref_col=self.sample_train.columns.tolist()[-1], place='after')


        return self.sample_train



    def save(self):
        """
        Save data file with new features 
        """

        #Save data as csv
        if self.csv == True:
            self.sample_train.to_csv(os.path.join(self.path_to_save, r'engineered_features_' + self.target + '.csv'))

            print('Result saved as: engineered_features_' + self.target + '.csv')
        
        # self.sample_train['source'] = 'train'
        # self.sample_test['source'] = 'test'
        # df_combined = pd.concat([self.sample_train, self.sample_test], axis=0, ignore_index=True)

        # joblib.dump(df_combined, os.path.join(self.path_to_save, r'df_' + self.target + '_engineered_features.pkl'))

        # print('Combined Features saved as: df_' + self.target + '_engineered_features.pkl')


        joblib.dump(self.new_cols, os.path.join(self.path_to_save, r'features_' + self.target + '_engineered.pkl'))

        print('Result saved as: features_' + self.target + '_engineered.pkl')

        # # Split dataset
        # df_train, df_test = train_test_split(self.sample_train, test_size=0.2, random_state=42) 
        
        joblib.dump(self.sample_train, os.path.join(self.path_to_save, r'df_train_' + self.target + '_engineered.pkl'))
        print('Train DataFrame saved as: df_train_' + self.target + '_engineered.pkl')

        
        # Filter the test dataframe where 'is_experimental' is 1.0
        if (self.sample_test['is_experimental'] == 1.0).any():
            test_df = self.sample_test[self.sample_test['is_experimental'] == 0.0]
            joblib.dump(test_df, os.path.join(self.path_to_save, r'df_test_' + self.target + '_engineered.pkl'))
            print('Test DataFrame saved as: df_test_expt_' + self.target + '_engineered.pkl')
            
            expt_df = self.sample_test[self.sample_test['is_experimental'] == 1.0]
            joblib.dump(expt_df, os.path.join(self.path_to_save, r'df_test_expt_' + self.target + '_engineered.pkl'))
            print('Test Expt DataFrame saved as: df_test_expt_' + self.target + '_engineered.pkl')
            
        else:
            joblib.dump(self.sample_test, os.path.join(self.path_to_save, r'df_test_' + self.target + '_engineered.pkl'))
            print('Test DataFrame saved as: df_test_' + self.target + '_engineered.pkl')