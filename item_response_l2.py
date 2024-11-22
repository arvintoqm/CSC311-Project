from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    question_id = data['question_id']
    user_id = data['user_id']
    is_correct = data['is_correct']
    log_lklihood = 0.0

    for i in range(len(question_id)):
        question = question_id[i]
        user = user_id[i]
        c_ij = is_correct[i]

        theta_i = theta[user]
        beta_j = beta[question]

        #p_correct = sigmoid(theta_i - beta_j)
        #log_lklihood += c_ij * np.log(p_correct) - (1 - c_ij) * np.log(1 - p_correct)

        log_lklihood += c_ij * (theta_i - beta_j) - np.log(1+np.exp(theta_i - beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, w_theta, w_beta, lambda_theta, lambda_beta):
    """Perform gradient descent step with weighted regularization.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: Learning rate
    :param theta: Vector of user abilities
    :param beta: Vector of question difficulties
    :param w_theta: Weights for regularization of theta
    :param w_beta: Weights for regularization of beta
    :return: Updated theta, beta
    """
    # Initialize gradients
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    user_id = np.array(data['user_id'])
    question_id = np.array(data['question_id'])
    is_correct = np.array(data['is_correct'], dtype=np.float64)

    prob_correct = sigmoid(theta[user_id] - beta[question_id])

    # Loop through the data to compute gradients
    for i in range(len(question_id)):
        q = question_id[i]
        u = user_id[i]
        c_ij = is_correct[i]

        grad_theta[u] += (c_ij - prob_correct[u])
        grad_beta[q] += -(c_ij - prob_correct[u])

    # Add regularization terms
    grad_theta -= lambda_theta * w_theta * theta  # Regularization for theta
    grad_beta -= lambda_beta * w_beta * beta     # Regularization for beta

    # Update parameters using gradient descent
    theta += lr * grad_theta
    beta += lr * grad_beta

    return theta, beta



def irt(data, val_data, lr, iterations, w_theta, w_beta, lambda_theta, lambda_beta):
    """
    Train IRT model with weighted regularization.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: Learning rate (float)
    :param iterations: Number of iterations (int)
    :param w_theta: Regularization weights for theta (1D numpy array)
    :param w_beta: Regularization weights for beta (1D numpy array)
    :param lambda_theta: Overall regularization weight for theta (float)
    :param lambda_beta: Overall regularization weight for beta (float)
    :return: theta, beta, val_acc_lst
    """
    question_id = data['question_id']
    user_id = data['user_id']
    theta = np.zeros(max(user_id) + 1)
    beta = np.zeros(max(question_id) + 1)

    val_acc_lst = []
    train_llk_lst = []
    val_llk_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk_lst.append(neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llk_lst.append(val_neg_lld)

        test_score = evaluate(data=data, theta=theta, beta=beta)

        valid_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(valid_score)

        print("NLLK: {} \t Test Score: {} \t Valid Score: {}".format(neg_lld, test_score, valid_score))
        theta, beta = update_theta_beta(data, lr, theta, beta, w_theta, w_beta, lambda_theta, lambda_beta)

    return theta, beta, val_acc_lst, train_llk_lst, val_llk_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.0001
    iterations = 60

    N, M = sparse_matrix.shape

    # Uniform weights as a starting point
    w_theta = np.random.randn(N) * 0.01
    w_beta = np.random.randn(M) * 0.01

    # Apply custom weights, if needed
    #w_theta[0:sparse_matrix.shape[0]] = 1  # Penalize first 10 users less
    # w_beta[100:300] = 2.0  # Penalize questions 250-300 more

    lambda_theta = 0.001
    lambda_beta = 0.001

    theta, beta, val_acc_lst, train_llk_lst, val_llk_lst = irt(train_data, val_data, learning_rate, iterations, w_theta, w_beta, lambda_theta, lambda_beta)

    valid_accuracy = evaluate(val_data, theta, beta)
    print(f"Final Test Accuracy: {valid_accuracy:.4f}")
    test_accuracy = evaluate(test_data, theta, beta)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    ### For training the data

    #plt.figure(figsize=(8, 6))
    #plt.plot(range(1, iterations + 1), val_acc_lst, marker='o', color="orange", label="Validation Accuracy")
    #plt.xlabel("Iteration")
    #plt.ylabel("Validation Accuracy")
    #plt.title("Validation Accuracy vs. Iterations")
    #plt.legend()
    #plt.grid()
    #plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    selected_questions = [1525, 1011, 1771] #, 82, 991
    #alphas = [0.8, 0.3, 0.1]
    #a = 0

    #plt.figure(figsize=(8, 6))
    #for j in selected_questions:
    #    beta_j = beta[j]
    #    probabilities = 1 / (1 + np.exp(-(theta_range - beta_j)))
    #    plt.plot(theta_range, probabilities, label=f"Question {j} (β={beta_j:.2f})",
    #             alpha=alphas[a])
    #    a += 1

    #beta_j1 = beta[selected_questions[0]]
    #beta_j2 = beta[selected_questions[1]]
    #beta_j3 = beta[selected_questions[2]]

    #prob_j1 = sigmoid(theta - beta_j1)
    #prob_j2 = sigmoid(theta - beta_j2)
    #prob_j3 = sigmoid(theta - beta_j3)

    #plt.scatter(theta, prob_j1, alpha=0.6, label=f"Question {selected_questions[0]} (β={beta_j1:.2f})", color='blue')
    #plt.scatter(theta, prob_j2, alpha=0.6, label=f"Question {beta[selected_questions[1]]} (β={beta_j2:.2f})", color='orange')
    #plt.scatter(theta, prob_j3, alpha=0.6, label=f"Question {selected_questions[2]} (β={beta_j3:.2f})", color='green')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
