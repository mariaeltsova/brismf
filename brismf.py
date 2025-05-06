import numpy as np

class BRISMF():                     
    def __init__(self, n_users, n_items,
                 n_factors, lr, reg, seed=42):

        rng = np.random.default_rng(seed)
       
        self.P = rng.uniform(-0.01, 0.01, (n_users, n_factors))
        self.Q = rng.uniform(-0.01, 0.01, (n_factors, n_items))
        self.lr, self.reg = lr, reg

        self.bias_idx_user = 0                     # workflow-bias column
        self.bias_idx_item = 1                     # icon-bias    row

        self.P[:, self.bias_idx_user] = 1.0        # keep them 1 (bias)
        self.Q[self.bias_idx_item, :] = 1.0
        
    #  Algorithm 1 from the paper (with the bias step)
    def fit(self,
            train,
            val_data = None,
            n_epochs = 50):
        best_rmse, epochs_no_improve = np.inf, 0
        
        for epoch in range(n_epochs):
           
            np.random.shuffle(train)
            for u, i, r in train:                
                pred = self.predict(u, i)
                err = r - pred

                self.P[u] += self.lr * (err * self.Q[:, i] - self.reg * self.P[u])
                self.Q[:, i] += self.lr * (err * self.P[u] - self.reg * self.Q[:, i])

                # Reset bias entries to fixed values
                self.P[u, 0] = 1.0
                self.Q[1, i] = 1.0
                
            # ---- validation & early-stop ------------------------------------
            if val_data is not None:
                rmse = self._rmse(val_data)
                if rmse < best_rmse:          
                    best_rmse, epochs_no_improve = rmse, 0
                    best_P, best_Q = self.P.copy(), self.Q.copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= 2:
                        break                     
        if val_data is not None:
            self.P, self.Q = best_P, best_Q

    def predict(self, u, i):
        return self.P[u] @ self.Q[:, i]
    
    def _rmse(self, data):
        se = sum((r - self.predict(u, i)) ** 2 for u, i, r in data)
        return np.sqrt(se / len(data))
        
    """
    Compute a latent vector for a new workflow by updating only p_u.
    Implements Algorithm 2
    """
    def fold_in_user(self, user_ratings, n_epochs=8):
        rng = np.random.default_rng()
        p = rng.uniform(-0.02, 0.02, self.P.shape[1])   
        for _ in range(n_epochs):
            for i, r in user_ratings:                  
                pred = p @ self.Q[:, i]
                err  = r - pred

                p += self.lr * (err * self.Q[:, i] - self.reg * p)  
                
        return p
    
    def fold_in_user_vector(self, rating_vector, n_epochs=8):
        idx = np.nonzero(rating_vector)[0]
        pairs = [(i, float(rating_vector[i])) for i in idx]
        return self.fold_in_user(pairs, n_epochs=n_epochs)


model = BRISMF(train_matrix_np.shape[0], train_matrix_np.shape[1], n_factors=n_factors, lr=0.003, reg=0.002)
model.fit(train_data, test_data)

"""
evaluate on the test_data modification where for each workflow each icon is removed once 
and the average reconstructed position is calculated
"""