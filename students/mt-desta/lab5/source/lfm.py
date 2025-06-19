import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
import random
from surprise import AlgoBase, PredictionImpossible
from surprise.utils import get_rng
from surprise.accuracy import rmse

class SVD(AlgoBase):
    """SVD algorithm for collaborative filtering.
    
    This implementation follows scikit-surprise's SVD algorithm style.
    
    Parameters
    ----------
    n_factors : int, default=100
        The number of factors.
        
    n_epochs : int, default=20
        The number of iteration of the SGD procedure.
        
    lr_all : float, default=0.005
        The learning rate for all parameters.
        
    reg_all : float, default=0.02
        The regularization term for all parameters.
        
    lr_bu : float, default=None
        The learning rate for user biases. If None, lr_all is used.
        
    lr_bi : float, default=None
        The learning rate for item biases. If None, lr_all is used.
        
    lr_pu : float, default=None
        The learning rate for user factors. If None, lr_all is used.
        
    lr_qi : float, default=None
        The learning rate for item factors. If None, lr_all is used.
        
    reg_bu : float, default=None
        The regularization term for user biases. If None, reg_all is used.
        
    reg_bi : float, default=None
        The regularization term for item biases. If None, reg_all is used.
        
    reg_pu : float, default=None
        The regularization term for user factors. If None, reg_all is used.
        
    reg_qi : float, default=None
        The regularization term for item factors. If None, reg_all is used.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for reproducibility.
        
    Attributes
    ----------
    pu_ : ndarray of shape (n_users, n_factors)
        User factors.
        
    qi_ : ndarray of shape (n_items, n_factors)
        Item factors.
        
    bu_ : ndarray of shape (n_users,)
        User biases.
        
    bi_ : ndarray of shape (n_items,)
        Item biases.
        
    global_mean_ : float
        The mean of all ratings.
    """
    
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        lr_bu: Optional[float] = None,
        lr_bi: Optional[float] = None,
        lr_pu: Optional[float] = None,
        lr_qi: Optional[float] = None,
        reg_bu: Optional[float] = None,
        reg_bi: Optional[float] = None,
        reg_pu: Optional[float] = None,
        reg_qi: Optional[float] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        
    def _initialize_factors(self) -> None:
        """Initialize the latent factor matrices and bias terms."""
        rng = get_rng(self.random_state)
        
        self.pu_ = rng.normal(0, 0.1, (self.trainset.n_users, self.n_factors))
        self.qi_ = rng.normal(0, 0.1, (self.trainset.n_items, self.n_factors))
        self.bu_ = np.zeros(self.trainset.n_users)
        self.bi_ = np.zeros(self.trainset.n_items)
        self.global_mean_ = self.trainset.global_mean
        
    def fit(self, trainset) -> 'SVD':
        """Fit the model to the training set.
        
        Parameters
        ----------
        trainset : Trainset
            The training set.
            
        Returns
        -------
        self : object
            Returns self.
        """
        AlgoBase.fit(self, trainset)
        self._initialize_factors()
        
        rng = get_rng(self.random_state)
        
        for epoch in range(self.n_epochs):
            # Shuffle the ratings
            indices = rng.permutation(len(self.trainset.ur))
            
            for u in indices:
                for i, r in self.trainset.ur[u]:
                    # Compute current prediction
                    pred = (
                        self.global_mean_ +
                        self.bu_[u] +
                        self.bi_[i] +
                        np.dot(self.pu_[u], self.qi_[i])
                    )
                    
                    # Compute error
                    err = r - pred
                    
                    # Update biases
                    self.bu_[u] += self.lr_bu * (err - self.reg_bu * self.bu_[u])
                    self.bi_[i] += self.lr_bi * (err - self.reg_bi * self.bi_[i])
                    
                    # Update factors
                    self.pu_[u] += self.lr_pu * (
                        err * self.qi_[i] - self.reg_pu * self.pu_[u]
                    )
                    self.qi_[i] += self.lr_qi * (
                        err * self.pu_[u] - self.reg_qi * self.qi_[i]
                    )
                    
                
        return self
    
    def estimate(self, u: int, i: int) -> float:
        """Estimate the rating for a user-item pair.
        
        Parameters
        ----------
        u : int
            The user ID.
            
        i : int
            The item ID.
            
        Returns
        -------
        float
            The estimated rating.
        """
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
            
        return (
            self.global_mean_ +
            self.bu_[u] +
            self.bi_[i] +
            np.dot(self.pu_[u], self.qi_[i])
        )
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-N recommendations for a user.
        
        Parameters
        ----------
        user_id : int
            ID of the user.
            
        n_recommendations : int, default=10
            Number of recommendations to return.
            
        Returns
        -------
        item_ids : ndarray of shape (n_recommendations,)
            IDs of recommended items.
            
        scores : ndarray of shape (n_recommendations,)
            Predicted scores for recommended items.
        """
        if not self.trainset.knows_user(user_id):
            raise PredictionImpossible('User is unknown.')
            
        user_predictions = (
            self.global_mean_ +
            self.bu_[user_id] +
            self.bi_ +
            np.dot(self.pu_[user_id], self.qi_.T)
        )
        
        # Get top-N items
        top_n_indices = np.argsort(user_predictions)[-n_recommendations:][::-1]
        top_n_scores = user_predictions[top_n_indices]
        
        return top_n_indices, top_n_scores
