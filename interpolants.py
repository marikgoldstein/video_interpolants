import torch

class LinearInterpolant:

    def __init__(self,):
        pass

    def compute_xt(self, t, x0, x1):
        return self.wide(self.alpha(t)) * x0 + self.wide(self.beta(t)) * x1

    def compute_xdot(self, t, x0, x1);
        return self.wide(self.alpha_dot(t)) * x0 + self.wide(self.beta_dot(t)) * x1

    def wide(self, t):
        return t[:, None, None, None]

    def alpha(self, t):
        return 1 - t

    def alpha_dot(self, t):
        return -1.0 * torch.ones_like(t)

    def beta(self, t):
        return t

    def beta_dot(self, t):
        return torch.ones_like(t)

class OurInterpolant:

    def __init__(self,):
        self.noise_strength = 0.1

    def compute_xt(self, t, x0, x1, noise):
        return self.wide(self.alpha(t)) * x0 + self.wide(self.beta(t)) * x1 + self.wide(self.gamma(t)) * noise

    def compute_xdot(self, t, x0, x1, noise);
        return self.wide(self.alpha_dot(t)) * x0 + self.wide(self.beta_dot(t)) * x1 * self.wide(self.sigma_dot(t) * t.sqrt()) * noise

    def wide(self, t):
        return t[:, None, None, None]

    def alpha(self, t):
        return 1 - t

    def alpha_dot(self, t):
        return -1.0 * torch.ones_like(t)

    def beta(self, t):
        return t.pow(2)

    def beta_dot(self, t):
        return 2.0 * t

    def sigma(self, t):
        return self.noise_strength * (1-t)

    def sigma_dot(self, t):
        return -1.0 * self.noise_strength * torch.ones_like(t)

    def gamma(self, t):
        return self.sigma(t) * t.sqrt()

