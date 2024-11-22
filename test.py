import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=50, hidden_size=25):
        super(AutoEncoder, self).__init__()
        self.encoder_1 = nn.Linear(num_question, k)
        self.encoder_2 = nn.Linear(k, hidden_size)

        self.decoder_1 = nn.Linear(hidden_size, k)
        self.decoder_2 = nn.Linear(k, num_question)

        self.dropout = nn.Dropout(0.2)

    def get_weight_norm(self):
        norms = [
            torch.norm(self.encoder_1.weight, 2) ** 2,
            torch.norm(self.encoder_2.weight, 2) ** 2,
            torch.norm(self.decoder_1.weight, 2) ** 2,
            torch.norm(self.decoder_2.weight, 2) ** 2,
        ]
        return sum(norms)

    def forward(self, inputs):
        encoded_1 = F.sigmoid(self.encoder_1(inputs))
        encoded_2 = F.sigmoid(self.encoder_2(self.dropout(encoded_1)))

        decoded_1 = F.sigmoid(self.decoder_1(encoded_2))
        output = F.sigmoid(self.decoder_2(decoded_1))

        return output




def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    training_loss = []
    val_acc = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0) + ((lamb/2) * model.get_weight_norm())

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        training_loss.append(train_loss)
        val_acc.append(valid_acc)

        # print(
        #     "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
        #         epoch, train_loss, valid_acc
        #     )
        # )

    return training_loss, val_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    k_star = 100
    model = None

    lr = 0.01
    num_epoch = 50
    lamb = 0.0001
    hidden_size = 50

    model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=k_star, hidden_size=hidden_size)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    valid_acc = evaluate(model, zero_train_matrix, valid_data)

    # Hyperparameter search space
    # k_values = [20, 50, 100]
    # hidden_sizes = [10, 25, 50] 
    # lamb_values = [0.0001, 0.001, 0.01, 0.1]
    # lr_values = [0.001, 0.005, 0.01] 

    # num_epoch = 50  
    # best_model = None
    # best_val_acc = 0
    # best_hyperparams = {}

    # results = []

    # for k in k_values:
    #     for hidden_size in hidden_sizes:
    #         for lamb in lamb_values:
    #             for lr in lr_values:
    #                 print(f"Testing: k={k}, hidden_size={hidden_size}, lamb={lamb}, lr={lr}")

    #                 model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=k, hidden_size=hidden_size)

    #                 train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    #                 valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #                 results.append((k, hidden_size, lamb, lr, valid_acc))

    #                 print(f"Validation Accuracy: {valid_acc:.4f}")

    #                 if valid_acc > best_val_acc:
    #                     best_val_acc = valid_acc
    #                     best_model = model
    #                     best_hyperparams = {
    #                         "k": k,
    #                         "hidden_size": hidden_size,
    #                         "lamb": lamb,
    #                         "lr": lr,
    #                     }

    # print(f"\nBest Hyperparameters: {best_hyperparams}")
    # print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy with Best Hyperparameters: {test_acc:.4f}")




if __name__ == "__main__":
    main()
