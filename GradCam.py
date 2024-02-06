# Define the hooks
activation = {}
gradient = {}

def forward_hook(module, input, output):
    activation['value'] = output

def backward_hook(module, grad_input, grad_output):
    gradient['value'] = grad_output[0]

# L2CNN
backward_hook_handle = score.layers[-1].register_backward_hook(backward_hook)
forward_hook_handle = score.layers[-1].register_forward_hook(forward_hook)
score(output, target).backward()

#Eff
backward_hook_handle = score.efficientnet._blocks[-1].register_backward_hook(backward_hook)
forward_hook_handle = score.efficientnet._blocks[-1].register_forward_hook(forward_hook)
score(imEst2).backward()
