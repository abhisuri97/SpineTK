from sklearn.model_selection import train_test_split
import pandas as pd

def make_split(baseline_directory, strat_var=None):
  with open(f'{baseline_directory}/metadata.json') as meta_file:
    df = pd.read_json(meta_file, orient='index')
    y = strat_var
    X_train_p, X_test, y_train_p, y_test = train_test_split(df, y, test_size=0.2, 
                                                      stratify=y, random_state=100, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_p, y_train_p, test_size=(10/70), 
                                                      stratify=y_train_p, random_state=100, shuffle=True)

  # Quick hack. But we should expect that all these proportions are roughly the same
  print(sum(y_train)/len(y_train), 'num: ', len(y_train))
  print(sum(y_val)/len(y_val), 'num: ', len(y_val))
  print(sum(y_test)/len(y_test), 'num: ', len(y_test))
  return X_train, X_val, X_test, y_train, y_val, y_test