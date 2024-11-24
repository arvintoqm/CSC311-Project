# Import necessary libraries
import numpy as np
import pandas as pd
from utils import load_train_csv, load_valid_csv, load_public_test_csv, evaluate, save_private_test_csv
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_data = load_train_csv("./data")
valid_data = load_valid_csv("./data")
test_data = load_public_test_csv("./data")

# Load metadata
question_meta = pd.read_csv("./data/question_meta.csv")
student_meta = pd.read_csv("./data/student_meta.csv")
subject_meta = pd.read_csv("./data/subject_meta.csv")

# Preprocess student metadata
current_year = 2024
student_meta['date_of_birth'] = pd.to_datetime(student_meta['data_of_birth'], errors='coerce')
student_meta['age'] = current_year - student_meta['date_of_birth'].dt.year
student_meta['age'] = student_meta['age'].fillna(student_meta['age'].median())
student_meta['premium_pupil'] = student_meta['premium_pupil'].fillna(0)

# Load pre-trained Word2Vec embeddings (GloVe vectors with 100 dimensions)
word2vec_model = api.load("glove-wiki-gigaword-100")

# Process subject IDs
def parse_subject_ids(subject_ids):
    subject_ids = subject_ids.strip("[]").split(", ")
    return [int(sid) for sid in subject_ids]

question_meta['subject_ids'] = question_meta['subject_id'].apply(parse_subject_ids)

# Create a mapping from subject_id to its name
subject_dict = subject_meta.set_index('subject_id')['name'].to_dict()

# Function to get the embedding for a subject name
def get_subject_embedding(name):
    words = name.lower().split()
    embeddings = [word2vec_model[word] for word in words if word in word2vec_model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word2vec_model.vector_size)

# Get embeddings for each subject
subject_embeddings = {sid: get_subject_embedding(name) for sid, name in subject_dict.items()}

# Function to get the mean embedding of all subjects associated with a question
def get_question_embedding(subject_ids):
    embeddings = [subject_embeddings[sid] for sid in subject_ids if sid in subject_embeddings]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word2vec_model.vector_size)

question_meta['embedding'] = question_meta['subject_ids'].apply(get_question_embedding)

# Stack all embeddings into a matrix and reduce dimensionality using PCA
embedding_matrix = np.vstack(question_meta['embedding'].values)
pca = PCA(n_components=1)
reduced_embeddings = pca.fit_transform(embedding_matrix)
question_meta['reduced_embedding'] = list(reduced_embeddings)

# Convert datasets to DataFrame
train_df = pd.DataFrame(train_data)
valid_df = pd.DataFrame(valid_data)
test_df = pd.DataFrame(test_data)

# Merge student features
train_df = train_df.merge(student_meta, on='user_id', how='left')
valid_df = valid_df.merge(student_meta, on='user_id', how='left')
test_df = test_df.merge(student_meta, on='user_id', how='left')

# Merge question features
question_features = question_meta[['question_id', 'reduced_embedding']]
train_df = train_df.merge(question_features, on='question_id', how='left')
valid_df = valid_df.merge(question_features, on='question_id', how='left')
test_df = test_df.merge(question_features, on='question_id', how='left')

# Fill missing embeddings with zeros
embedding_size = pca.n_components
zero_embedding = np.zeros(embedding_size)

train_df['reduced_embedding'] = train_df['reduced_embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_embedding)
valid_df['reduced_embedding'] = valid_df['reduced_embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_embedding)
test_df['reduced_embedding'] = test_df['reduced_embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_embedding)

# Calculate student average correctness and add as a feature
student_avg = train_df.groupby('user_id')['is_correct'].mean().reset_index()
student_avg.columns = ['user_id', 'student_avg_correct']

train_df = train_df.merge(student_avg, on='user_id', how='left')
valid_df = valid_df.merge(student_avg, on='user_id', how='left')
test_df = test_df.merge(student_avg, on='user_id', how='left')

global_avg_correct = train_df['is_correct'].mean()
train_df['student_avg_correct'] = train_df['student_avg_correct'].fillna(global_avg_correct)
valid_df['student_avg_correct'] = valid_df['student_avg_correct'].fillna(global_avg_correct)
test_df['student_avg_correct'] = test_df['student_avg_correct'].fillna(global_avg_correct)

# Prepare features
def preprocess_df(df):
    features = []
    for idx, row in df.iterrows():
        additional_features = np.array([
            row['age'],
            row['premium_pupil'],
            row['student_avg_correct']
        ])
        question_embedding = row['reduced_embedding']
        feature = np.concatenate((additional_features, question_embedding))
        features.append(feature)
    labels = df['is_correct'].values if 'is_correct' in df.columns else None
    features = np.array(features)
    return (features, labels) if labels is not None else features

train_features, train_labels = preprocess_df(train_df)
valid_features, valid_labels = preprocess_df(valid_df)
test_features = preprocess_df(test_df)

# -----------------
# 1. Ensemble Methods
# -----------------

# Train ensemble models
def train_ensemble_models(train_features, train_labels):
    models = {}
    # Logistic Regression Ensemble
    lr_models = []
    for _ in range(3):
        X_resampled, y_resampled = resample(train_features, train_labels)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_resampled, y_resampled)
        lr_models.append(model)
    models['logistic_regression'] = lr_models

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_features, train_labels)
    models['random_forest'] = rf_model

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100)
    gb_model.fit(train_features, train_labels)
    models['gradient_boosting'] = gb_model

    return models

def ensemble_predict(models, features):
    # Logistic Regression Ensemble
    lr_preds = np.mean([model.predict_proba(features)[:, 1] for model in models['logistic_regression']], axis=0)
    # Random Forest
    rf_preds = models['random_forest'].predict_proba(features)[:, 1]
    # Gradient Boosting
    gb_preds = models['gradient_boosting'].predict_proba(features)[:, 1]
    # Average predictions
    preds = (lr_preds + rf_preds + gb_preds) / 3
    return preds

ensemble_models = train_ensemble_models(train_features, train_labels)
ensemble_val_preds = ensemble_predict(ensemble_models, valid_features)
ensemble_val_accuracy = evaluate({'is_correct': valid_labels}, ensemble_val_preds)
print(f'Ensemble Validation Accuracy: {ensemble_val_accuracy:.4f}')

# -----------------
# 2. Item Response Theory (IRT)
# -----------------

# Implement IRT model (1PL model)
from scipy.optimize import minimize

def irt(train_data, val_data, lr=0.01, iterations=50):
    # Initialize parameters
    num_users = train_data['user_id'].nunique()
    num_questions = train_data['question_id'].nunique()

    user_ids = train_data['user_id'].unique()
    question_ids = train_data['question_id'].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    question_id_map = {qid: idx for idx, qid in enumerate(question_ids)}

    theta = np.zeros(num_users)  # user ability
    beta = np.zeros(num_questions)  # question difficulty

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Prepare data
    train_data['user_idx'] = train_data['user_id'].map(user_id_map)
    train_data['question_idx'] = train_data['question_id'].map(question_id_map)

    for iteration in range(iterations):
        # Compute gradients
        theta_grad = np.zeros(num_users)
        beta_grad = np.zeros(num_questions)
        for i, row in train_data.iterrows():
            u = int(row['user_idx'])
            q = int(row['question_idx'])
            c = row['is_correct']
            x = theta[u] - beta[q]
            p = sigmoid(x)
            theta_grad[u] += (c - p)
            beta_grad[q] += (p - c)
        # Update parameters
        theta += lr * theta_grad
        beta += lr * beta_grad

        # Compute validation accuracy
        val_acc = irt_evaluate(val_data, theta, beta, user_id_map, question_id_map)
        print(f'Iteration {iteration+1}, Validation Accuracy: {val_acc:.4f}')

    return theta, beta, user_id_map, question_id_map

def irt_evaluate(data, theta, beta, user_id_map, question_id_map):
    from sklearn.metrics import accuracy_score
    preds = []
    actual = []
    for i, row in data.iterrows():
        uid = row['user_id']
        qid = row['question_id']
        c = row['is_correct']
        if uid in user_id_map and qid in question_id_map:
            u = user_id_map[uid]
            q = question_id_map[qid]
            p = 1 / (1 + np.exp(-(theta[u] - beta[q])))
            preds.append(p >= 0.5)
            actual.append(c)
        else:
            # If user or question not in training data, predict average
            preds.append(0.5)
            actual.append(c)
    acc = accuracy_score(actual, preds)
    return acc

# Train IRT model
theta, beta, user_id_map, question_id_map = irt(train_df, valid_df, lr=0.01, iterations=10)

# Evaluate on validation set
irt_val_acc = irt_evaluate(valid_df, theta, beta, user_id_map, question_id_map)
print(f'IRT Validation Accuracy: {irt_val_acc:.4f}')

# -----------------
# 3. Neural Networks
# -----------------

# Implement a Neural Network using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Prepare data for PyTorch
X_train = torch.tensor(train_features, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32)
X_valid = torch.tensor(valid_features, dtype=torch.float32)
y_valid = torch.tensor(valid_labels, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# Define Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.out(self.fc2(x))
        return x

input_size = train_features.shape[1]
model = NeuralNet(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid).squeeze()
        val_preds = (val_outputs >= 0.5).float()
        val_acc = (val_preds == y_valid).float().mean()
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_acc:.4f}')

# -----------------
# 4. Matrix Factorization Model
# -----------------

# Implement Matrix Factorization
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Create user-question interaction matrix
def create_interaction_matrix(df):
    user_ids = df['user_id'].unique()
    question_ids = df['question_id'].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    question_id_map = {qid: idx for idx, qid in enumerate(question_ids)}

    num_users = len(user_ids)
    num_questions = len(question_ids)

    row_indices = df['user_id'].map(user_id_map).values
    col_indices = df['question_id'].map(question_id_map).values
    data = df['is_correct'].values

    interaction_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_questions))
    return interaction_matrix, user_id_map, question_id_map

interaction_matrix, mf_user_id_map, mf_question_id_map = create_interaction_matrix(train_df)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
svd.fit(interaction_matrix)

# Predict function
def mf_predict(data, svd, user_id_map, question_id_map):
    preds = []
    actual = []
    for i, row in data.iterrows():
        uid = row['user_id']
        qid = row['question_id']
        c = row['is_correct']
        if uid in user_id_map and qid in question_id_map:
            u_idx = user_id_map[uid]
            q_idx = question_id_map[qid]
            u_vec = svd.transform(interaction_matrix[u_idx])
            q_vec = svd.components_[:, q_idx]
            pred = np.dot(u_vec, q_vec)
            preds.append(pred >= 0.5)
            actual.append(c)
        else:
            preds.append(0.5)
            actual.append(c)
    acc = accuracy_score(actual, preds)
    return acc

# Evaluate on validation set
mf_val_acc = mf_predict(valid_df, svd, mf_user_id_map, mf_question_id_map)
print(f'Matrix Factorization Validation Accuracy: {mf_val_acc:.4f}')
