"""
Contains abstract functionality for learning locally linear sparse model.
"""
import copy
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.LIPEx_model = None
        self.LIME_model = None

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)
    
    # below is the new feature_selection method
    def feature_selection_new(self, data, labels, sel_labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1])) # return all features
        elif method == 'forward_selection':
            # labels: [num_samples, #classes]
            sel_features = []
            for label in sel_labels:
                sel_features += self.forward_selection(data, labels[:, label], weights, num_features).tolist()
            return np.unique(np.array(sel_features))
        elif method == 'auto':
            n_method = 'forward_selection'
            return self.feature_selection_new(data, labels, sel_labels, weights, num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   weights=None,
                                   used_features=None):
        if weights is None:
            weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        if used_features is None:
            used_features = self.feature_selection(neighborhood_data,
                                                    labels_column,
                                                    weights,
                                                    num_features,
                                                    feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        self.LIME_model = easy_model

        prediction_score = easy_model.score(     
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True), # sorted by decreasing absolute value of x[1]
                prediction_score, local_pred, used_features)
    
    # below is the new function
    def explain_instance_with_data_new(self,
                                    neighborhood_data,
                                    neighborhood_labels,
                                    distances,
                                    labels,
                                    num_features,
                                    feature_selection='auto',
                                    model_regressor=None,
                                    weights=None,
                                    used_features=None):
        """Takes perturbed data, labels and distances, returns explanation."""
        # labels: numpy array, index of labels by increasing order for which we want to use in feature selection
        alpha = 0.001
        learning_rate = 0.01
        if weights is None:
            weights = self.kernel_fn(distances)
        # labels_column = neighborhood_labels[:, label]
        if used_features is None:
            used_features = self.feature_selection_new(neighborhood_data,
                                                        neighborhood_labels,
                                                        labels,
                                                        weights,
                                                        num_features,
                                                        feature_selection)
        # train the explainable model
        if model_regressor is None:
            model_regressor_TVLoss = Explainer_g(len(used_features),len(neighborhood_labels[0,:]),loss_fn_name='TVLoss', alpha=alpha, learning_rate=learning_rate, max_epochs=1000, target_loss=0.001)
            model_regressor_HDLoss = Explainer_g(len(used_features),len(neighborhood_labels[0,:]),loss_fn_name='HDLoss', alpha=alpha, learning_rate=learning_rate, max_epochs=1000, target_loss=0.001)
        
        neighborhood_data = torch.from_numpy(neighborhood_data[:, used_features]).float()
        neighborhood_labels = torch.from_numpy(neighborhood_labels).float()
        weights = torch.from_numpy(weights).float()
       
        easy_model_TVLoss = model_regressor_TVLoss
        easy_model_HDLoss = model_regressor_HDLoss
        
        easy_model_TVLoss.fit(neighborhood_data, neighborhood_labels, sample_weight=weights)
        easy_model_HDLoss.fit(neighborhood_data, neighborhood_labels, sample_weight=weights)
        
        # plot the two loss curves
        # self.plot_loss_compare(easy_model_TVLoss, easy_model_HDLoss, alpha, learning_rate)

        if easy_model_HDLoss.best_loss < easy_model_TVLoss.best_loss:
            easy_model = easy_model_HDLoss
        else:
            easy_model = easy_model_TVLoss
        self.LIPEx_model = easy_model

        local_pred = easy_model.predict(neighborhood_data[0].reshape(1, -1))
        prediction_score = None
        if self.verbose:
            print('Right:','\n', torch.round(neighborhood_labels[0],decimals=5))
            print('Prediction_local:','\n', local_pred)
        return (easy_model.intercept,
                easy_model.coef,
                prediction_score, 
                local_pred, 
                used_features.tolist()) # from numpy to list
    
    def plot_loss_compare(self, TVLoss, HDLoss, alpha, learning_rate):
        plt.figure(figsize=(16,16))
        fig, ax = plt.subplots()
        ax.plot(TVLoss.training_loss, label='TVLoss')
        ax.plot(HDLoss.training_loss, label='HDLoss')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.yscale('log')
        plt.title("LearningRate= " + str(learning_rate) + "  " + " WeightDecay= " + str(alpha))
        plt.legend()
        PATH ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/plot/LIPEx_loss_curve/CompareLoss.png'
        plt.tight_layout()
        plt.savefig(PATH)
        plt.close()
        return
    
    def get_used_features(self,
                          neighborhood_data,
                          neighborhood_labels,
                          distances,
                          label,
                          num_features,
                          feature_selection='auto'):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.

            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.

            score is the R^2 value of the returned explanation

            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(     
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


import torch.nn as nn
import torch
torch.set_printoptions(precision=5, sci_mode=False)
from collections import OrderedDict

class CustomLoss(nn.Module):
    def __init__(self, loss_name=None) -> None:
        super(CustomLoss, self).__init__()
        self.loss_name = loss_name
    
    def forward(self, output, target, weights):
        # output: (num_samples, num_classes)
        # target: (num_samples, num_classes)
        # weights: (num_samples,)
        if self.loss_name == 'TVLoss':
            return self.TV_Loss(output, target, weights)
        else:
            return self.HD_Loss(output, target, weights)
    
    def TV_Loss(self, output, target, weights):
        TotalVar_Loss = torch.mean(0.5 * (weights*torch.sum(torch.abs(output-target), dim=-1))) # recommend threshold is 0.1
        return TotalVar_Loss
    
    def HD_Loss(self, output, target, weights):
        Hellinger_Loss = (0.5 * (weights * torch.sum((torch.sqrt(output) - torch.sqrt(target))**2,dim=-1)).sum())/output.shape[0]
        return Hellinger_Loss


class Explainer_g(nn.Module):
    def __init__(self, input_size, num_classes, loss_fn_name=None, alpha=0.01, learning_rate=0.01,max_epochs=1000, target_loss=0.001):
        super(Explainer_g, self).__init__()
        torch.manual_seed(42) # for reproducibility
        self.net = nn.Sequential(
            OrderedDict({
                'linear': nn.Linear(input_size, num_classes, bias=True),
                'softmax': nn.Softmax(dim=1),
            })
        )
        # print('Explainer_g is initialized with number of trainable parametes: ', input_size*num_classes+num_classes)
        
        # self.loss_fn = nn.CrossEntropyLoss(reduction='none') # (num_samples)
        self.loss_fn = CustomLoss(loss_name = loss_fn_name)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, weight_decay=alpha)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=alpha) # Use Adam instead of SGD
        self.training_loss = []
        self.intercept = None
        self.coef = None
        self.loss_fn_name = loss_fn_name
        self.weight_decay = alpha
        self.max_epochs = max_epochs
        self.batch_size = 128  
        self.target_loss = target_loss
        self.best_loss = torch.tensor(float('inf'))
        # print("initialized weight_matrix:",self.net.linear.weight.data)
    

    def fit(self, x, y, sample_weight):
        epoch, cnt = 0, 0

        loss = torch.tensor(1.0)
        if x.shape[0] % self.batch_size == 0:
            num_batches = x.shape[0] // self.batch_size
        else:
            num_batches = x.shape[0] // self.batch_size + 1
        # while loss > 0.05 and epoch < 1000 and cnt < 100:
        while epoch < self.max_epochs:
            self.net.train()
            for i in range(num_batches):
                batch_x = x[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y[i*self.batch_size:(i+1)*self.batch_size]
                batch_sample_weight = sample_weight[i*self.batch_size:(i+1)*self.batch_size]
                self.optimizer.zero_grad()
                out = self.net(batch_x)
                loss = self.loss_fn(out, batch_y, batch_sample_weight)
                self.training_loss.append(loss.item())
                # print ('Step [%d], Training Loss: %.4f' %(len(self.training_loss), loss.data))
                if loss < self.target_loss:
                    epoch = self.max_epochs
                    print("Break here (Target_loss 0.001 satisfied).")
                    print ('%s: Step [%d], Training Loss: %.4f' %(self.loss_fn_name, len(self.training_loss), loss.data))
                    break
                loss.backward()
                self.optimizer.step()
                if loss < self.best_loss:
                    cnt = 0
                    self.best_loss = loss 
                    best_model = copy.deepcopy(self.net.state_dict())
                elif cnt >1000:
                    self.net.load_state_dict(best_model)
                    epoch = self.max_epochs
                    print("Break here. (cnt > 1000)")
                    print ('%s: Step [%d], Training Loss: %.4f' %(self.loss_fn_name, len(self.training_loss), loss.data))
                    break
                else:
                    cnt += 1
            epoch += 1
        self.coef, self.intercept = self.net.linear.weight.data.detach().numpy(), self.net.linear.bias.data.detach().numpy()
        # print("weight_matrix:",self.net.linear.weight.data)
        return
    
    def predict(self, x):
        self.net.eval()
        return self.net(x)
    
    def plot_loss(self):
        plt.plot(self.training_loss)
        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.yscale('log')
        plt.title('Training Loss vs Step ' + self.loss_fn_name + " alpha: " + str(self.weight_decay))
        plt.axhline(y=self.target_loss, color='r', linestyle='-',label='target_loss 0.001')
        plt.legend()
        PATH = 'LIPEx/code_Img/Plot/'+self.loss_fn_name+'.png'
        # plt.show()
        plt.savefig(PATH)
        plt.close()
        return


    
    