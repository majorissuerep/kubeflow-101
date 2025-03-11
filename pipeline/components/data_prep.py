import kfp
from kfp.components import OutputPath, create_component_from_func

def data_prep(
    data_path: str,
    x_train_path: OutputPath(str),
    y_train_path: OutputPath(str),
    x_test_path: OutputPath(str),
    y_test_path: OutputPath(str),
    test_size: float = 0.2,
    random_state: int = 42
):
    """Prepare data for training by splitting into train and test sets."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Assume the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save the data splits
    X_train.to_csv(x_train_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

# Create KFP component
data_prep_op = create_component_from_func(
    func=data_prep,
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
) 