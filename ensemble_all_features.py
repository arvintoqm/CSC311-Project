import numpy as np
import pandas as pd
from utils import load_train_csv, load_valid_csv, load_public_test_csv, evaluate, save_private_test_csv
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.decomposition import PCA
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')

# Load the main datasets
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

# Convert train_data to DataFrame
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

# Prepare features with user_id and question_id one-hot encoding
def preprocess_df(df, num_users, num_questions):
    features = []
    for idx, row in df.iterrows():
        user_feat = np.zeros(num_users)
        question_feat = np.zeros(num_questions)
        user_feat[int(row['user_id'])] = 1
        question_feat[int(row['question_id'])] = 1
        additional_features = np.array([
            row['age'],
            row['premium_pupil'],
            row['student_avg_correct']
        ])
        question_embedding = row['reduced_embedding']
        feature = np.concatenate((user_feat, question_feat, additional_features, question_embedding))
        features.append(feature)
    labels = df['is_correct'].values if 'is_correct' in df.columns else None
    features = np.array(features)
    return (features, labels) if labels is not None else features

num_users = train_df['user_id'].nunique()
num_questions = train_df['question_id'].nunique()

train_features, train_labels = preprocess_df(train_df, num_users, num_questions)
valid_features, valid_labels = preprocess_df(valid_df, num_users, num_questions)
test_features = preprocess_df(test_df, num_users, num_questions)

# Train ensemble model
def train_ensemble(train_features, train_labels, n_estimators=10):
    models = []
    for _ in range(n_estimators):
        X_resampled, y_resampled = resample(train_features, train_labels)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_resampled, y_resampled)
        models.append(model)
    return models

def ensemble_predict(models, features):
    preds = np.mean([model.predict_proba(features)[:, 1] for model in models], axis=0)
    return preds

models = train_ensemble(train_features, train_labels, n_estimators=3)
val_preds = ensemble_predict(models, valid_features)
val_accuracy = evaluate({'is_correct': valid_labels}, val_preds)
print(f'Validation Accuracy: {val_accuracy:.4f}')
