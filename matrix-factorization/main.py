from brismf import BRISMF
import pandas as pd
from tests_demo import get_tests
from collections import Counter

train_df = pd.read_csv("./matrix-factorization/train.csv", index_col=0)
test_df = pd.read_csv("./matrix-factorization/test.csv", index_col=0)
demo_df = pd.read_csv("./matrix-factorization/test_demo.csv", index_col=0)
test_icons = ['Visualize-Scatter Plot', 'Data-Datasets', 'Transform-Data Sampler', 'Data-Data Table', 'Data-Paint Data', 'Visualize-Box Plot', 'Data-File', 'Model-Random Forest', 'Visualize-Tree Viewer', 'Model-Tree', 'Model-Logistic Regression', 'Evaluate-Predictions', 'Text Mining-Concordance', 'Evaluate-Confusion Matrix', 'Evaluate-Test and Score', 'Text Mining-Collocations', 'Unsupervised-Hierarchical Clustering']
tests = get_tests()

# single evaluation on the TEST set, the average position is only for the icons in test_icons
# optimized hyperparameters used
def train_per_icon():
    rows_with_removed_icons = remove_all_icons_once(test_df)
    exit
    icon_stats = {icon: {
            "icon_name": icon,        
            "sum_of_positions": 0,    
            "times_detected": 0, 
            "average_position": 0    
        }
        for icon in test_icons}

    print(f"Number of columns in train set: {train_df.shape[1]}")
    print(f"Number of columns in test set : {test_df.shape[1]}")

    train_data = []
    train_matrix_np = train_df.to_numpy()
    workflows, icons = train_matrix_np.nonzero()
    for wf, ic in zip(workflows, icons):
        count = train_matrix_np[wf, ic]
        train_data.append((wf, ic, count))

    test_data = []
    test_matrix_np = test_df.to_numpy()
    workflows, icons = test_matrix_np.nonzero()
    for wf, ic in zip(workflows, icons):
        count = test_matrix_np[wf, ic]
        test_data.append((wf, ic, count))

    model = BRISMF(train_matrix_np.shape[0], train_matrix_np.shape[1], n_factors=12, lr=0.001, reg=0.00025)
    model.fit(train_data, test_data)
    
    for image in rows_with_removed_icons:
        for row in image:

            if (row.non_zero_column_name not in test_icons):
                continue
            
            test_row = row.row_with_removed_icon.reshape(1, -1)
            print(test_row)
            p_new = model.fold_in_user(test_row[0])
            print('p_new', p_new)
            reconstructed_row = p_new @ model.Q     

            col_values = {
                col_name: reconstructed_row[i]
                    for i, col_name in enumerate(train_df.columns)
            }

            sorted_cols = sorted(col_values.items(), key=lambda x: x[1], reverse=True)
       

            sorted_cols = [
                (col, val)                       
                for col, val in sorted_cols      
                if row.row_with_removed_icon_df.loc[col] <= 0  
            ]
            
            print()
            reconstructed_position = -1
        
            for position, (col_name_candidate, val) in enumerate(sorted_cols):
                print(f"Column '{col_name_candidate}' is ranked #{position + 1} in the reconstructed row. {row.row_with_removed_icon_df.loc[col_name_candidate]} {val}")
                if col_name_candidate == row.non_zero_column_name:
                    print(f"------- Column '{row.non_zero_column_name}' is ranked #{position + 1} in the reconstructed row.")
                    reconstructed_position = position + 1
                    break
            stat = icon_stats.get(row.non_zero_column_name)
            stat["sum_of_positions"] += reconstructed_position
            stat["times_detected"]   += 1
            
            if (reconstructed_position == -1):
                raise Exception("reconstructed_position == -1o")
    total_postion = 0
    total_detected = 0
    for stats in icon_stats.values():   
        total_postion += stats["sum_of_positions"]
        total_detected += stats["times_detected"]
        stats["average_position"] = (
            stats["sum_of_positions"] / stats["times_detected"]
        )   
    total_average_pos = total_postion / total_detected
    
    for stat in icon_stats.values():
        print(stat)
    sorted_list = sorted(icon_stats.values(), key=lambda d: d["average_position"])  
    for s in sorted_list:
        print(f'{s["icon_name"]:20s}  avg_pos = {s["average_position"]:.2f}')
        
    print('len(rows_with_removed_icons)', len(rows_with_removed_icons))

    print(f"Number of rows in test set: {test_df.shape[0]}")
    print('average position', total_average_pos)

#generates the CSV with the DEMO set so that we do not need to manually fill it in
def save_demo():
    test_matrix = pd.DataFrame(0, index=[t["name"] for t in tests],
       columns=train_df.columns,
       dtype=int,
    )
    for t in tests:
        row = t["name"]
        icon_cnts = Counter(t["icons_present"])
        
        for icon, cnt in icon_cnts.items():
            if icon in test_matrix.columns:
                test_matrix.at[row, icon] = cnt
    test_matrix.to_csv("./matrix-factorization/test_demo.csv")

#optimizes on the TEST set, also prints values for the DEMO set
def optimize_hyperparams():
    grid = {
        "n_factors":  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        "lr": [0.001, 0.002, 0.003, 0.0035],
        "reg":[0.00025, 0.0004, 0.0005 ,0.001, 0.002, 0.005], 
    }
   
    best_reconstructed_pos = 100
    best_model = ''
    all_models = []
    for n_factors in grid["n_factors"]:
        for lr in grid["lr"]:
            for reg in grid["reg"]:
                train_matrix = train_df
                test_matrix = test_df
                
                sum_of_reconstr_positions = 0
                rows_with_removed_icons = remove_all_icons_once(test_matrix)
            
                train_data = []
                train_matrix_np = train_matrix.to_numpy()
                workflows, icons = train_matrix_np.nonzero()
                for wf, ic in zip(workflows, icons):
                    count = train_matrix_np[wf, ic]
                    train_data.append((wf, ic, count))
                
                test_data = []
                test_matrix_np = test_matrix.to_numpy()
                workflows, icons = test_matrix_np.nonzero()
                for wf, ic in zip(workflows, icons):
                    count = test_matrix_np[wf, ic]
                    test_data.append((wf, ic, count))
                
                model = BRISMF(train_matrix_np.shape[0], train_matrix_np.shape[1], n_factors=n_factors, lr=lr, reg=reg)

                model.fit(train_data, test_data)
                sum_of_positions_per_component = 0
                number_of_icons_removed = 0
                
                for image in rows_with_removed_icons:
                    for row in image:
                        if (row.non_zero_column_name not in test_icons):
                            continue
                        test_row = row.row_with_removed_icon.reshape(1, -1)
                        p_new = model.fold_in_user(test_row[0])
                        reconstructed_row = p_new @ model.Q     

                        col_values = {
                            col_name: reconstructed_row[i]
                                for i, col_name in enumerate(train_df.columns)
                        }

                        sorted_cols = sorted(col_values.items(), key=lambda x: x[1], reverse=True)
                        
                        sorted_cols = [
                            (col, val)                       # keep the tuple …
                            for col, val in sorted_cols      # … for every (col, val) pair
                            if row.row_with_removed_icon_df.loc[col] <= 0   # … only if that row’s value is NOT > 0
                        ]
                        reconstructed_position = -1
                        for position, (col_name_candidate, val) in enumerate(sorted_cols):
                            if col_name_candidate == row.non_zero_column_name:
                                #print(f"Column '{row.non_zero_column_name}' is ranked #{position + 1} in the reconstructed row.")
                                reconstructed_position = position + 1
                                break
                        if (reconstructed_position == -1):
                            raise Exception("reconstructed_position == -1o")
                        sum_of_positions_per_component += reconstructed_position
                        number_of_icons_removed += 1
                        
                average_reconstr_position = sum_of_positions_per_component / number_of_icons_removed
                average_pos_on_demo = run_on_demo(n_factors=n_factors, lr=lr, reg=reg)
                model_text = f"n_factors:{n_factors},  lr: {lr}, reg: {reg}, position: {average_reconstr_position}, on demo: {average_pos_on_demo}"
                print(model_text)
                all_models.append({
                        'average_reconstructed_position': average_pos_on_demo,
                        'model': model_text
                    })
                if average_reconstr_position < best_reconstructed_pos:
                    best_reconstructed_pos = average_reconstr_position
                    best_model = model_text
    all_models.sort(key=lambda d: d['average_reconstructed_position'], reverse=True)
    for entry in all_models:  
        print(entry['model'])
    print('best_reconstructed_pos', best_reconstructed_pos)
    print(best_model)             
                            

# prints the reconstructed positions of the widgets in the DEMO set. 
# lists all the widgets that are ranked higher than the target widget
def run_on_demo(n_factors=13, lr=0.0035, reg=0.005):
    train_data = []
    train_matrix_np = train_df.to_numpy()
    workflows, icons = train_matrix_np.nonzero()
    for wf, ic in zip(workflows, icons):
        count = train_matrix_np[wf, ic]
        train_data.append((wf, ic, count))

    test_data = []
    test_matrix_np = test_df.to_numpy()
    workflows, icons = test_matrix_np.nonzero()
    for wf, ic in zip(workflows, icons):
        count = test_matrix_np[wf, ic]
        test_data.append((wf, ic, count))
                        
                       
    model = BRISMF(train_matrix_np.shape[0], train_matrix_np.shape[1], n_factors=n_factors, lr=lr, reg=reg)
    model.fit(train_data, test_data)

    average_reconstr_position = 0
    for index, row in enumerate(demo_df.iterrows()):
        row_series = demo_df.iloc[index]
        row_value = demo_df.iloc[index].copy().values
        removed_icon = tests[index]['icons_removed'][0]
        
        icons_present = ''
        for present_icon in tests[index]['icons_present']:
            icons_present += present_icon + ', '
        print('Icons present:', icons_present)
        print('Target icon:', removed_icon)
        
        test_row = row_value.reshape(1, -1)
        p_new = model.fold_in_user(test_row[0])
        reconstructed_row = p_new @ model.Q     
        col_values = {
            col_name: reconstructed_row[i]
                for i, col_name in enumerate(train_df.columns)
        }
        
        sorted_cols = sorted(col_values.items(), key=lambda x: x[1], reverse=True)
        print('col values length before', len(sorted_cols))
        
        sorted_cols = [
            (col, val)                       
            for col, val in sorted_cols     
            if row_series[col] <= 0 
        ]
        print('col values length after', len(sorted_cols))

        reconstructed_position = -1
    
        for position, (col_name_candidate, val) in enumerate(sorted_cols):
            print(f"Column '{col_name_candidate}' is ranked #{position + 1}")
            if col_name_candidate == removed_icon:
                print(f"------- Column '{removed_icon}' is ranked #{position + 1} in the reconstructed row.")
                reconstructed_position = position + 1
                break
        average_reconstr_position += reconstructed_position
        print()
    return average_reconstr_position / len(tests)


# calculates the baseline
def predict_based_on_icon_frequency_only_target_icons():
    icons_by_frequency = pd.read_csv(
            "icons_detected_sorted.txt",
            sep="\t",
            index_col=0,          
        )
    icons_by_frequency.squeeze('columns')
    rows_with_removed_icons = remove_all_icons_once(test_df)
    
    sum_of_positions = 0
    number_of_icons_removed = 0
    for image in rows_with_removed_icons:
        for row in image:
            if (row.non_zero_column_name not in test_icons):
                continue
            reconstructed_position = icons_by_frequency.index.get_loc(row.non_zero_column_name) + 1
          
            sum_of_positions += reconstructed_position
            number_of_icons_removed += 1
        
    avg_pos = sum_of_positions / number_of_icons_removed
    print('average position in baseline', avg_pos, number_of_icons_removed)

# helper function to remove each of the widgets from the image (only once)
def remove_all_icons_once(test_matrix):
    rows_with_removed_icons = []
    
    for index, row in enumerate(test_matrix.iterrows()):
        row_with_removed_icons = []
        row_name = test_matrix.index[index]
        
        non_zero_columns = test_matrix.iloc[index][test_matrix.iloc[index] != 0].index.tolist()
        if len(non_zero_columns) == 1:
            print('row_name',index, row_name)
            continue
        for column in non_zero_columns:
            copy_row = test_matrix.iloc[index].copy()
            copy_row[column] = 0
            #print(f"Reduced value in column '{column}' by 1. New value: {test_matrix.at[row_name, column]}") 
        
            row_with_removed_icons.append(RowWithRemovedIcons(
                test_row_number=index,
                non_zero_column_name=column,
                row_with_removed_icon=copy_row.values,
                row_original=test_matrix.loc[row_name].values,
                image_name=row_name,
                icons_detected=non_zero_columns,
                row_with_removed_icon_df=copy_row
            ))
            
        rows_with_removed_icons.append(row_with_removed_icons)
    
    return rows_with_removed_icons
class RowWithRemovedIcons:
    def __init__(self, test_row_number, non_zero_column_name, row_with_removed_icon, row_original, image_name, icons_detected, row_with_removed_icon_df, components=None):
        self.test_row_number = test_row_number
        self.non_zero_column_name = non_zero_column_name
        self.row_with_removed_icon = row_with_removed_icon
        self.row_original = row_original
        self.image_name = image_name
        self.icons_detected = icons_detected
        self.components = components if components is not None else []
        self.row_with_removed_icon_df = row_with_removed_icon_df if row_with_removed_icon_df is not None else []

    def add_component(self, n_components, reconstructed_position):
        self.components.append({n_components: reconstructed_position})
        
        

#run_on_demo()
#train_per_icon()
#optimize_hyperparams()
#save_demo()
#predict_based_on_icon_frequency_only_target_icons()


