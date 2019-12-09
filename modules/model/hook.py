class Hook():
    '''
    A basic hook class that stores information for each of the intermediary
    modules of a given pytorch model. We use this in order to isolate the
    outputs of individual layers in the BERT model (i.e. attention scores).
    '''
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.name = module._get_name()
    def close(self):
        self.hook.remove()
