import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO


data = torch.zeros(10)
data[0:6] = 1.0

# def original_model(data):
#     f = pyro.sample("latent_fairness", dist.Beta(10.0, 10.0))
#     print(f)
#     with pyro.plate("data", data.size(0)):
#         pyro.sample("obs", dist.Bernoulli(f), obs=data)


def train(model, guide, lr=0.01):
    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    n_steps = 101
    for step in range(n_steps):
        loss = svi.step(data)
        if step % 50 == 0:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))

def model_mle(data):
    # note that we need to include the interval constraint;
    # in original_model() this constraint appears implicitly in
    # the support of the Beta distribution.
    f = pyro.param("latent_fairness", torch.tensor(0.5),
                   constraint=constraints.unit_interval)
    with pyro.plate("data", data.size(0)):
        pyro.sample("obs", dist.Bernoulli(f), obs=data)


def guide_mle(data):
    pass


train(model_mle, guide_mle)


print("Our MLE estimate of the latent fairness is {:.3f}".format(
      pyro.param("latent_fairness").item()))