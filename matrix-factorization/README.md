## Datasets

All datasets are mutually exclusive.

- **`train`** – train dataset (331 cases)  
- **`test`** – test dataset; the average reconstructed position is based on this set (65 cases)  
- **`demo`** – the set of "images" manually selected to demonstrate in the text of the thesis (12 cases).  
  New demo cases can be added by adding a new test case to `tests_demo.py` and running `save_demo()`.  
  Here, `icons_removed` contains just one icon – it is not present in the image, and we want to see what its reconstructed position would be.

## Files

- **`brismf.py`** – class of the model, functions to train it and to predict new values  
- **`main.py`** – functions to train the model on `train` data, optimize hyperparameters, and create the demo output  
- **`tests_demo.py`** – a list of test cases. Used as described above.