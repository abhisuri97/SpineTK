import get_dicts

def return_dicts(tr_type, train, val, test, baseline_directory):
  if tr_type == 'train':
    return get_dicts(baseline_directory, train.index.to_list(), "train")
  if tr_type == 'val':
    return get_dicts(baseline_directory, val.index.to_list(), "val")
  if tr_type == 'test':
    return get_dicts(baseline_directory, test.index.to_list(), "test")