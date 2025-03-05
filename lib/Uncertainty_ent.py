import numpy as np
from sklearn.metrics import log_loss


def model_uncertainty(model, x_test, x_train, y_train, unc_method="bays", laplace_smoothing=1, log=False):
    
    if "bays" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, laplace_smoothing)
        porb_matrix = get_prob(model, x_test, laplace_smoothing)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_ent_bays(porb_matrix, likelyhoods)
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for RF")

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty

############################################################################################

def uncertainty_ent_bays(probs, likelihoods): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	# print("prob\n", probs)
	# print("likelihoods in bays", likelihoods)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	# print("entropy\n", entropy)

	a = np.sum(entropy, axis=2)
	al = a * likelihoods
	a = np.sum(al, axis=1)
   
	given_axis = 1
	dim_array = np.ones((1,probs.ndim),int).ravel()
	dim_array[given_axis] = -1
	b_reshaped = likelihoods.reshape(dim_array)
	mult_out = probs*b_reshaped
	p_m = np.sum(mult_out, axis=1)

	# p_m = np.mean(p, axis=1) #* likelihoods

	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a

def get_likelyhood(model_ens, x_train, y_train, laplace_smoothing, a=0, b=0, log=False):
    likelyhoods  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob_train = estimator.predict_proba(x_train) 
        else:
            tree_prob_train = tree_laplace_corr(estimator,x_train, laplace_smoothing, a, b)

        likelyhoods.append(log_loss(y_train,tree_prob_train))
    likelyhoods = np.array(likelyhoods)
    likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
    likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

    if log:
        print(f"<log>----------------------------------------[]")
        print(f"likelyhoods = {likelyhoods}")
    return np.array(likelyhoods)

def get_prob(model_ens, x_data, laplace_smoothing, a=0, b=0, log=False):
    prob_matrix  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob = estimator.predict_proba(x_data) 
        else:
            tree_prob = tree_laplace_corr(estimator,x_data, laplace_smoothing,a,b)
        prob_matrix.append(tree_prob)
    if log:
        print(f"<log>----------------------------------------[]")
        print(f"prob_matrix = {prob_matrix}")
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix

def tree_laplace_corr(tree, x_data, laplace_smoothing, a=0, b=0):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            if a != 0 or b != 0:
                if i==0:
                    L = a
                else:
                    L = b
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob