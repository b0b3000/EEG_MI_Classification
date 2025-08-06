#Old function for conditional selection of training instances

def accept_confident_probabilities(probs, k, Y_test):
    # Get indices of sorted probabilities (descending)
    sorted_indices = np.argsort(-probs, axis=1)

    # Highest and second-highest probabilities
    top1_probs = probs[np.arange(len(probs)), sorted_indices[:, 0]]
    top2_probs = probs[np.arange(len(probs)), sorted_indices[:, 1]]

    # Apply condition: only keep prediction if top1 >= 2 * top2
    preds = sorted_indices[:, 0]
    mask = top1_probs >= k * top2_probs
    print(len(mask))

    made_preds = preds[mask]
    Y_test_new = Y_test[mask]
    return made_preds, Y_test_new, mask