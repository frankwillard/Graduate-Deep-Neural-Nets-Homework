#ATTACKS
import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start:
      x_perturbed = x_nat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    else:
      x_perturbed = x_nat.clone().detach()

    # Make sure the sample is projected into original distribution bounds [0,1]
    #x_perturbed = (x_perturbed - torch.min(x_perturbed)) / (torch.max(x_perturbed) - torch.min(x_perturbed))
    x_perturbed = torch.clip(x_perturbed.clone().detach(), min = 0, max = 1)

    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        current_grad = gradient_wrt_data(model, device, x_perturbed, lbl)

        # Perturb the image using the gradient
        gradient_step = alpha * torch.sign(current_grad)
        x_perturbed_unclipped = x_perturbed.clone().detach() + gradient_step

        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        # Total difference between pixel and original never greater than epsilon
        #clipped_gradient_step = torch.clip(gradient_step, min = -eps, max = eps)
        clipped_differences = torch.clip(x_perturbed_unclipped.clone().detach() - x_nat, min = -eps, max = eps)
        x_perturbed = x_nat.clone().detach() + clipped_differences

        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x_perturbed = torch.clip(x_perturbed.clone().detach(), min = 0, max = 1)

    # Return the final perturbed samples
    return x_perturbed


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: FGSM is a special case of PGD

    #PGD_attack(model, device, dat, lbl, eps, alpha = eps, iters, rand_start)

    return PGD_attack(model, device, dat, lbl, eps, alpha = eps, iters = 1, rand_start = False)


def rFGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: rFGSM is a special case of PGD

    return PGD_attack(model, device, dat, lbl, eps, alpha = eps, iters = 1, rand_start = True)


def FGM_L2_attack(model, device, dat, lbl, eps):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # Compute gradient w.r.t. data
    current_grad = gradient_wrt_data(model, device, x_nat, lbl)

    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    # HINT: Flatten gradient tensor first, then compute L2 norm
    #l2_of_grad = torch.norm(current_grad)
    
    l2_of_grad = torch.Tensor([torch.norm(torch.flatten(samp)) for samp in current_grad])

    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
    clamped_grad = torch.clip(l2_of_grad, min=1e-12)

    unit_grad = [(samp / clamp).detach().cpu().numpy()  for samp, clamp in zip(current_grad, clamped_grad)]

    # Add perturbation the data
    x_perturbed = x_nat.clone().detach() + eps * torch.Tensor(unit_grad).to(device)

    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_perturbed = torch.clip(x_perturbed.clone().detach(), min = 0, max = 1)

    # Return the perturbed samples
    return x_perturbed