# For General Kalman Filtering

import numpy as np

class KalmanFilter:

    def __init__(self, A, B, C, initial_state):
        self._A = A
        self._B = B
        self._C = C

        self._state_dimension = np.shape(self._A)[0]
        self._initial_state = initial_state
        self._current_state = None
        self._current_covariance_matrix = None
        self._predicted_state = None
        self._predicted_covariance_matrix = None

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def C(self):
        return self._C

    @property
    def state_dimension(self):
        return self._state_dimension

    def update_predicted_state(self, current_input):
        x_predicted = self._A * self._current_state + self._B * current_input
        self._predicted_state = x_predicted

    def update_predicted_covariance_matrix(self):
        sigma_predicted = self._A * self._current_covariance_matrix * np.transpose(self._A) + self._R
        self._predicted_covariance_matrix = sigma_predicted

    def get_kalman_gain(self):
        T = self._C * self._predicted_covariance_matrix * np.transpose(self._C) + self._Q
        K = self._predicted_covariance_matrix * np.transpose(self._C) * np.inv(T)
        self._kalman_gain = K

    def get_next_state(self, current_measurement):
        x_new = self._predicted_state + \
                self._kalman_gain * (current_measurement - self._C * self._predicted_state)
        return x_new

    def get_next_covariance_matrix(self):
        I = np.eye(self._state_dimension)
        sigma_new = (I - self._kalman_gain * self._C) * self._predicted_covariance_matrix
        return sigma_new


    # Get mean and covariance matrix of initial state
    def get_initial_state(self):
        self._current_state = self._initial_state['mean']
        self._current_covariance_matrix = self._initial_state['covariance']

    # Get covariance matrix for noise of state and measurement
    def get_noise(self, R, Q):
        self._R = R
        self._Q = Q


    def do_kalman_filter(self, input, measurement, R, Q):
        self.get_noise(R, Q)
        self.update_predicted_state(input)
        self.update_predicted_covariance_matrix()

        state_filtered = self.get_next_state(measurement)
        covariance_matrix_filtered = self.get_next_covariance_matrix()

        self._current_state = state_filtered
        self._current_covariance_matrix = covariance_matrix_filtered

    def get_current_state(self):
        return self._current_state

    def get_current_covariance_matrix(self):
        return  self._current_covariance_matrix


class AdaptiveKalmanFilter(KalmanFilter):

    def __init__(self, A, B, C, initial_state, number_window):
        super().__init__(A, B, C, initial_state)

        self._number_window = number_window
        self._innovation_count = 0
        self._innovation = None
        self._covariance_innovation = None
        self._R = None
        self._Q = None
    @property
    def number_window(self):
        return self._number_window

    def update_covariance_innovation(self):
        C_k = 0
        for i in range(self._innovation_count):
            C_k += np.dot(self._innovation[i], self._innovation[i])
        self._covariance_innovation = C_k / self._innovation_count

    def do_adaptive_kalman_filter(self, input, measurement):

        # ToDO : To define initial R and Q
        if self._innovation_count == 0:
            self.do_kalman_filter(input, measurement, self._R, self._Q)
            current_innovation = measurement - self._C * self._predicted_state
            np.append(self._innovation, current_innovation)
            self._innovation_count += 1

        else:
            self.update_covariance_innovation()
            self._R = self._covariance_innovation - \
                              self._C * self._predicted_covariance_matrix * np.transpose(self._C)
            self._Q = self._kalman_gain * self._covariance_innovation * np.transpose(self._kalman_gain)

            self.do_kalman_filter(input, measurement, self._R, self._Q)

            current_innovation = measurement - self._C * self._predicted_state
            np.append(self._innovation, current_innovation)

            if self._innovation_count == self._number_window:
                self._innovation = self._innovation[, 1:]
            else:
                self._innovation_count += 1







