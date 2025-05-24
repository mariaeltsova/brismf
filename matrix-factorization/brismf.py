import numpy as np

class BRISMF():                     
    def __init__(self, n_users, n_items,
                 n_factors=40, lr=0.01, reg=0.02, seed=42):

        rng = np.random.default_rng(seed)
       
        self.P = rng.uniform(-0.01, 0.01, (n_users, n_factors))
        self.Q = rng.uniform(-0.01, 0.01, (n_factors, n_items))
        self.lr, self.reg = lr, reg

        self.bias_idx_user = 0                    
        self.bias_idx_item = 1                     

        self.P[:, self.bias_idx_user] = 1.0       
        self.Q[self.bias_idx_item, :] = 1.0
        
    
    def fit(self,
            train,
            val_data = None,
            n_epochs = 10):
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

    def predict(self, u, i) -> float:
        return self.P[u] @ self.Q[:, i]
    
    def _rmse(self, data) -> float:
        se = sum((r - self.predict(u, i)) ** 2 for u, i, r in data)
        return np.sqrt(se / len(data))
        
    def fold_in_user(self,
                 user_ratings,
                 n_steps: int = 100,
                 seed: int = 7) -> np.ndarray:
        user_ratings = [(idx, float(count))
                for idx, count in enumerate(user_ratings) if count > 0]

        rng  = np.random.default_rng()
        p  = rng.uniform(-0.01, 0.01, self.P.shape[1])


        p[self.bias_idx_user] = 1.0                
        p[self.bias_idx_item] = 0.0              

        for _ in range(n_steps):
            for i, r in user_ratings:
                pred = p @ self.Q[:, i]
                err  = r - pred
                p += self.lr * (err * self.Q[:, i] - self.reg * p)
                p[self.bias_idx_user] = 1.0

        return p