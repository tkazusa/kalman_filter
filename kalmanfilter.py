# coding=utf-8

# write code...

import numpy as np
import math

class klf(object):
    def __init__(self):
        self.T = None
        self.y = None
        self.x0 = None
        self.Sigma0 = None
        self.F = None
        self.G = None
        self.Q = None
        self.H = None
        self.R = None
        self.exo_var = None
        self.num_exo_var = None


    def set_state_transition_model(self, F, G, Q):
        self.F = F
        self.G = G
        self.Q = Q

    def set_observation_model(self, H, R, num_exogenous_variables):
        self.H = H
        self.R = R
        self.num_exo_var = num_exogenous_variables


    def filtering(self, T, y, x0, Sigma0, exogenous_variables):
        self.T = T
        self.y = y
        self.x0 = x0
        self.Sigma0 = Sigma0
        self.exo_var = exogenous_variables

        #Initiate values
        x = self.x0
        Sigma = self.Sigma0


        #Initial predicted values
        M = x
        x_estimated = x[0].T
        y_estimated = [self.H.dot(x)]
        ei_values = [0]
        S_values = [self.R]

        for time in range(self.T):
            #estimation
            x_ = self._estimate_state(x, time, self.exo_var, self.num_exo_var)
            Sigma_ = self._estimate_statecov(Sigma)

            if math.isnan(self.y[time+1]):
                x = x_
                Sigma = Sigma_
                y_ = self._get_estimated_y(x)

            else:
                #update prams
                ei = self._get_obsereved_error(x_, time)
                S = self._get_observed_errorcov(Sigma_)
                K = self._get_kalmangain(Sigma_, S)
                x = self._update_state(x_, K, ei)
                Sigma = self._update_error_mat(Sigma_, K)
                y_ = self._get_estimated_y(x)

            M = np.c_[M, x]
            x_estimated = np.r_[x_estimated, x[0].T]
            y_estimated.append(y_)
            ei_values.append(ei)
            S_values.append(S)

        M = M.T
        y_estimated = np.array(y_estimated)
        ei_values = np.array(ei_values)
        S_values = np.array(S_values)

        return M, x_estimated, y_estimated, ei_values, S_values


    def _estimate_state(self, x, time, exo_var, num_exo_var):
        x_ = self.F.dot(x)
        x_[-1*num_exo_var:] = exo_var[time+1]
        return x_


    def _estimate_statecov(self, Sigma):
        return self.F.dot(Sigma).dot(self.F.T) + self.G.dot(self.Q).dot(self.G.T)

    def _get_obsereved_error(self, x_, time):
        return self.y[time + 1] - self.H.dot(x_)

    def _get_observed_errorcov(self, Sigma_):
        return self.H.dot(Sigma_).dot(self.H.T) + self.R

    def _get_kalmangain(self, Sigma_, S):
        if S.ndim == 0:
            K = Sigma_.dot(self.H.T) * 1 / S
        else:
            K = Sigma_.dot(self.H.T).dot(np.linalg.inv(S))
        return K

    def _update_state(self, x_, K, ei):
        return x_ + K.dot(ei)

    def _update_error_mat(self, Sigma_, K):
        temp_mat = K.dot(self.H)
        if temp_mat.ndim == 0:
            Sigma = Sigma_ - temp_mat * Sigma_
        else:
            Sigma = Sigma_ - temp_mat.dot(Sigma_)
        return Sigma

    def _get_estimated_y(self, x):
        return self.H.dot(x)

    def get_prediction(self, x):
        x_ = self.F.dot(x)
        y_ = self.H.dot(x_)
        return x_, y_





