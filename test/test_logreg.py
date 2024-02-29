"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import LogisticRegressor, utils
from sklearn.preprocessing import StandardScaler
import numpy as np

# Common function for loading dataset
@pytest.fixture
def logistic_regressor_and_data():
    
    # Load data
    features = [
        'Penicillin V Potassium 500 MG',
        #'Computed tomography of chest and abdomen',
        #'Plain chest X-ray (procedure)',
        #'Low Density Lipoprotein Cholesterol',
        #'Creatinine',
        'AGE_DIAGNOSIS'
    ]
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=features,
        split_percent=0.8,
        split_seed=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize the Logistic Regressor with the number of features 
    model = LogisticRegressor(num_feats=len(features) - 1)
    
    return model, X_train_scaled, X_val_scaled, y_train, y_val


# Test for checking if make_prediction function is working correctly 
def test_prediction(logistic_regressor_and_data):

    # Load dataset
    model, X_train, X_val, y_train, y_val = logistic_regressor_and_data
    
    # Padding data with vector of ones for bias term
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    # Generate predictions for the validation set
    predictions = model.make_prediction(X_val)
	
    # Check that all predictions are between 0 and 1
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    # Check that the shape of the predictions matches the number of samples in X_val
    assert predictions.shape[0] == X_val.shape[0]
    

# Test for checking loss function
def test_loss_function(logistic_regressor_and_data):
    
    # Load dataset
    model, X_train, X_val, y_train, y_val = logistic_regressor_and_data
    
    # Padding data with vector of ones for bias term
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
    # Generate predictions for the validation set
    predictions = model.make_prediction(X_val)
    
    # Calculate loss
    loss = model.loss_function(y_val, predictions)

    # Check that loss values are positive
    assert loss >= 0
    
    # Test case 1: No loss
    y_true_1 = np.array([1])
    y_pred_1 = np.array([1])
    expected_loss_1 = 0
    calculated_loss_1 = model.loss_function(y_true_1, y_pred_1)
    # Checking that loss value is expected loss value
    assert np.isclose(calculated_loss_1, expected_loss_1)

    # Test case 2: non-zero loss
    y_true_2 = np.array([0])
    y_pred_2 = np.array([0.5])
    expected_loss_2 = -np.log(0.5)
    calculated_loss_2 = model.loss_function(y_true_2, y_pred_2)
    # Checking that loss value is expected loss value
    assert np.isclose(calculated_loss_2, expected_loss_2)

# Checking if gradient function is working correctly
def test_gradient(logistic_regressor_and_data):
    
    # Test case with single feature and data
    X = np.array([[1, 1]])  # Feature matrix with bias term
    y = np.array([0])    # True label
    
    model = LogisticRegressor(num_feats=1)
    model.W = np.array([0.5, -0.5])  # Initialize weights
    
    # Predict probabilities
    y_pred = model.make_prediction(X)
    
    # Calculate expected gradient
    expected_gradient = np.dot(X.T, (y_pred - y)) / len(y) 
    
    # Calculate the gradient using model
    calculated_gradient = model.calculate_gradient(y, X)
    
    # Check if calculated gradient matches the expected gradient
    assert np.allclose(calculated_gradient, expected_gradient)



# Test for checking if the weights update after training
def test_training(logistic_regressor_and_data):
    # Load data
    model, X_train, X_val, y_train, y_val = logistic_regressor_and_data

    # Record the initial weights for comparison
    initial_weights = np.copy(model.W)

    # Train the model on the dataset
    model.train_model(X_train, y_train, X_train, y_train)  # Using X_train as both train and validation for simplicity

    # Check if the weights have been updated
    assert not np.array_equal(initial_weights, model.W)