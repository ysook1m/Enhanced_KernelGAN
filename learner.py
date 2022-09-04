class Learner:
    # Default hyper-parameters
    lambda_update_freq = 66
    bic_loss_to_start_change = 2.0
    lmd_bicubic_decay_rate = 100.
    update_l_rate_freq = 250
    update_l_rate_rate = 10.
    lambda_sparse_end = 5
    lambda_centralized_end = 1
    lmd_bicubic_min = 5e-6

    def __init__(self):
        self.bic_loss_counter = 0
        self.similar_to_bicubic = False  # Flag indicating when the bicubic similarity is achieved
        self.insert_constraints = True  # Flag is switched to false once constraints are added to the loss

    def update(self, iteration, gan):
        if iteration == 0:
            return
        # Update learning rate every update_l_rate freq
        if gan.lmd_fakeblur > 0.0:
            if iteration >= 1000:
                if iteration % 100 == 0:
                    gan.lmd_fakeblur = gan.lmd_fakeblur - 0.125
                    if gan.lmd_fakeblur < 0.0:
                        gan.lmd_fakeblur = 0.0
            
        if iteration > 2000:
            if iteration % self.update_l_rate_freq == 0:
                for params in gan.optimizer_G.param_groups:
                    params['lr'] /= self.update_l_rate_rate
                # for params in gan.optimizer_D.param_groups:
                    # params['lr'] /= self.update_l_rate_rate
                for params in gan.optimizer_D2.param_groups:
                    params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if gan.l_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif iteration % self.lambda_update_freq == 0 and gan.lmd_bicubic > self.lmd_bicubic_min:
            gan.lmd_bicubic = max(gan.lmd_bicubic / self.lmd_bicubic_decay_rate, self.lmd_bicubic_min)
            if self.insert_constraints and gan.lmd_bicubic < 5e-3:
                gan.lambda_centralized = self.lambda_centralized_end
                gan.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False
