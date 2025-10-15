import torch

def get_interpolant(interpolant_type):

    if interpolant_type == 'linear':
        I = LinearInterpolant()
    elif interpolant_type == 'ours':
        I = OurInterpolant()
    else:
        assert False

    return I

class LinearInterpolant:

    def __init__(self,):
        pass

    def compute_xt(self, t, z0, z1):
        return self.wide(self.alpha(t)) * z0 + self.wide(self.beta(t)) * z1

    def compute_xdot(self, t, z0, z1):
        return self.wide(self.alpha_dot(t)) * z0 + self.wide(self.beta_dot(t)) * z1

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
        
    def info(self,):    
        # below, dot means time derivative.
        # z0 := Frame(t-1)
        # z1 := Frame(t)
        # noise := randn_like(z1)
        # z_t := a(t) z0 + b(t) z1 + sigma(t) root(t) noise
        # j is a random num less than t-1 
        # Reference Frame = Frame(t-1)
        # Context Frame = Frame(j)
        # Context frame makes the sampler not fully markovian.
        # cond = ( Frame(t-1), Frame(j), (t-1)-j) ) =  (Ref, Context, Gap) 
        # b_hat(z_t, t, cond) = E[drift_target | z_t] 
        # drift target is adot(t) z0 + bdot(z1 + sigmadot(t)root(t) noise.
        # note that the drift target isn't just the time derivative of the velocity
        # but rather the time derivative of velocity + coef * score
        # finally:
        # dZ^t_s = b_hat ds + sigma(s)dW_s          
        print("info on the interpolant")

    def compute_xt(self, t, z0, z1, noise):
        return self.wide(self.alpha(t)) * z0 + self.wide(self.beta(t)) * z1 + self.wide(self.gamma(t)) * noise

    def compute_drift_target(self, t, z0, z1, noise):
        a = self.wide(self.alpha_dot(t))
        b = self.wide(self.beta_dot(t))
        noise_coef = self.wide(self.sigma_dot(t) * t.sqrt())
        return a * z0 + b * z1 + noise_coef * noise

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

