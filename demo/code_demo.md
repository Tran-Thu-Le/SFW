```python
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
torch.manual_seed(123)

def inner_prod(u, v, k):
    return (u.T @ v)*k

def norm(u, k):
    return (k*u.pow(2).sum(0)).sqrt()

def gauss_comp(t, x, w, k):
    # x shape [m]
    # t shape [n]

    # compute ingredients
    t_copy = t.clone()
    t_copy.requires_grad = True
    diff = x.view(-1, 1)-t_copy
    z = diff**2/w**2 #shape [m, n]
    ln2 = torch.log(torch.tensor(2.))
    factor1 = (ln2*2./w**2)*diff #shape [m, n]
    factor2 = factor1**2 - (ln2*2/w**2)*t #shape [m, n]

    rawatom = torch.pow(2, -z) #shape [m, n]
    rawatom_deriv = factor1*rawatom
    rawatom_deriv2 = factor2*rawatom
    
    norms = norm(rawatom, k).view(-1)
    invnorm = 1./norms
    invnorm_deriv = grad(invnorm.sum(), t_copy, create_graph=True)[0]
    invnorm_deriv2 = grad(invnorm_deriv.sum(), t_copy, create_graph=True)[0]
    
    # detach step
    tensors = [rawatom, rawatom_deriv, rawatom_deriv2, invnorm, invnorm_deriv, invnorm_deriv2]
    tensors_new = []
    for tens in tensors:
        tens = tens.detach()
        tensors_new += [tens]
    rawatom, rawatom_deriv, rawatom_deriv2, invnorm, invnorm_deriv, invnorm_deriv2 = tensors_new


    # a_t = g_t*(1/n_t)
    atom = rawatom*invnorm
    # a_t' = g_t'*(1/n_t) + g_t*(1/n_t)'
    atom_deriv = rawatom_deriv*invnorm + rawatom*invnorm_deriv 
    # a_t'' = g_t''*(1/n_t) + 2*g_t'*(1/n_t)'+g_t*(1/n_t)''
    atom_deriv2 = rawatom_deriv2*invnorm + 2.*rawatom_deriv*invnorm_deriv + rawatom*invnorm_deriv2
    

    out = {
        "atom": atom,
        "atom_deriv": atom_deriv,
        "atom_deriv2": atom_deriv2,
    }
    return out

def run(w):
    """
    This function returns the Vanishing Derivative Pre-Certificate

    Setup:
        Omega = T = [0., 1.]
        Hilbert space: L^2(Omega)
        Parameter set: T 
        Gaussian kernel: a_t = k*2**(-(x-t)**2/w**2) where k is a L^2 normalized coeff

    Notations:
        Gamma = [a_t1, ..., a_tn,    (a_t1)', ..., (a_tn)']
        Gamma_star_inverse = Gamma * (Gamma_star * Gamma)

    Steps:
        Setup T, atom
        Compute y = c1*a_t1 + c2*a_t2 + c3*a_t3
        Compute Gamma_star_inverse
        Find minimal-norm dual feasible vector u = Gamma_star_inverse @ [sign(c), 0]
        Vanishing derivative pre-cert eta = <u, a_t>
    """

    # Setup
    a, b, n = 0., 1., 300
    k = torch.tensor((b-a)/n ) # coeff of integral inner_prod
    x = torch.linspace(a, b, n) # Omega in L^2(Omega)
    t = torch.tensor([0.1, 0.3, 0.95])
    c = torch.tensor([0.3, 0.3, 0.4]).view(-1, 1)
    out = gauss_comp(t, x, w, k)
    atom, atom_deriv, atom_deriv2 = out["atom"], out["atom_deriv"], out["atom_deriv2"]

    # Compute y
    y = atom @ c

    # Compute Gamma_star_inverse
    Gamma = torch.cat([atom, atom_deriv], 1)
    Gamma_deriv = torch.cat([atom_deriv, atom_deriv2], 1)
    Gamma_star_inverse = torch.inverse(inner_prod(Gamma, Gamma, k) )
    
    # Find u
    sign_and_zero = torch.cat([torch.sign(c).view(-1), torch.zeros(3)]).view(-1, 1)
    u = Gamma @ (Gamma_star_inverse @ sign_and_zero)


    # Show cert(t1, t2, t3)
    cert_val = inner_prod(u, atom, k).view(-1)
    cert_deriv = inner_prod(u, atom_deriv, k).view(-1)
    cert_deriv2 = inner_prod(u, atom_deriv2, k).view(-1)
    print(f"cert vals = {cert_val}")
    print(f"cert deriv = {cert_deriv}")
    print(f"cert deriv2 = {cert_deriv2}")

    # Plot cert
    t_cert = torch.linspace(0., 1., 201)
    out = gauss_comp(t=t_cert, x=x, w=w, k=k)
    atom_grid = out["atom"]
    eta = inner_prod(u, atom_grid, k).view(-1)
    plt.plot(t_cert, eta, label="vanishing deriv pre-cert", ls="--")
    plt.hlines(y=1., xmin=a, xmax=b)
    plt.vlines(x=t, ymin=0., ymax=1.)
    plt.legend()
    # plt.ylim(0.998, 1.002)
    plt.figure(figsize=(10,1))
    plt.show()



if __name__=="__main__":
    run(w=0.1)
```
