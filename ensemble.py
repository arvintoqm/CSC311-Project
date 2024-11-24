import numpy as np
from utils import *
from sklearn.linear_model import LogisticRegression

# use papers conda
# simple ensemble model 
def bootstrap_data(data, n_samples):
    """Generate a bootstrapped dataset from the original data."""
    indices = np.random.choice(len(data['user_id']), size=n_samples, replace=True)
    boot_data = {
        'user_id': [data['user_id'][i] for i in indices],
        'question_id': [data['question_id'][i] for i in indices],
        'is_correct': [data['is_correct'][i] for i in indices]
    }
    return boot_data

def train_base_model(data, num_users, num_questions):
    """Train a logistic regression model."""
    X = []
    y = data['is_correct']
    for uid, qid in zip(data['user_id'], data['question_id']):
        # One-hot encoding for user and question IDs
        user_feat = np.zeros(num_users)
        user_feat[uid] = 1
        question_feat = np.zeros(num_questions)
        question_feat[qid] = 1
        features = np.concatenate((user_feat, question_feat))
        X.append(features)
    X = np.array(X)
    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def predict(model, data, num_users, num_questions):
    """Generate predictions using the trained model."""
    X = []
    for uid, qid in zip(data['user_id'], data['question_id']):
        user_feat = np.zeros(num_users)
        user_feat[uid] = 1
        question_feat = np.zeros(num_questions)
        question_feat[qid] = 1
        features = np.concatenate((user_feat, question_feat))
        X.append(features)
    X = np.array(X)
    preds = model.predict_proba(X)[:, 1]  # Probability of correctness
    return preds

def main():
    # Load data
    train_data = load_train_csv("data")
    valid_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    num_users = max(train_data['user_id']) + 1
    num_questions = max(train_data['question_id']) + 1
    n_samples = len(train_data['user_id'])

    models = []
    for i in range(3):
        # Generate bootstrapped dataset
        boot_data = bootstrap_data(train_data, n_samples)
        # Train base model
        model = train_base_model(boot_data, num_users, num_questions)
        models.append(model)
        print(f"Model {i+1} trained.")

    # Generate predictions from each model
    val_preds = []
    test_preds = []
    for model in models:
        val_pred = predict(model, valid_data, num_users, num_questions)
        val_preds.append(val_pred)
        test_pred = predict(model, test_data, num_users, num_questions)
        test_preds.append(test_pred)

    # Average predictions
    val_preds_avg = np.mean(val_preds, axis=0)
    test_preds_avg = np.mean(test_preds, axis=0)

    # Evaluate performance
    val_accuracy = evaluate(valid_data, val_preds_avg)
    test_accuracy = evaluate(test_data, test_preds_avg)
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
