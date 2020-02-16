import math


def default_decay(num_epochs):
    step_epoch = min(num_epochs // 5, 50)

    def decay(epoch):
        if epoch < step_epoch:
            return 1e-4
        else:
            return 1e-5

    return decay


def cyclic_learning_rate(learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular'):
    """Applies cyclic learning rate (CLR).
       From the paper:
       Smith, Leslie N. "Cyclical learning
       rates for training neural networks." 2017.
       [https://arxiv.org/pdf/1506.01186.pdf]
        This method lets the learning rate cyclically
       vary between reasonable boundary values
       achieving improved classification accuracy and
       often in fewer iterations.
        This code varies the learning rate linearly between the
       minimum (learning_rate) and the maximum (max_lr).
        It returns the cyclic learning rate. It is computed as:
         ```python
         cycle = floor( 1 + global_step /
          ( 2 * step_size ) )
        x = abs( global_step / step_size – 2 * cycle + 1 )
        clr = learning_rate +
          ( max_lr – learning_rate ) * max( 0 , 1 - x )
         ```
        Polices:
          'triangular':
            Default, linearly increasing then linearly decreasing the
            learning rate at each cycle.
           'triangular2':
            The same as the triangular policy except the learning
            rate difference is cut in half at the end of each cycle.
            This means the learning rate difference drops after each cycle.
           'exp_range':
            The learning rate varies between the minimum and maximum
            boundaries and each boundary value declines by an exponential
            factor of: gamma^global_step.

    """

    def cyclic_lr(step):
        # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
        cycle = math.floor(1 + step / (2 * step_size))
        x = math.fabs(step / step_size - 2 * cycle + 1)
        clr = (max_lr - learning_rate) * max(0, 1 - x)

        if mode == 'triangular2':
            clr /= math.pow(2, (cycle - 1))
        elif mode == 'exp_range':
            clr *= math.pow(gamma, step)

        return clr + learning_rate

    return cyclic_lr
