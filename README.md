# Contents
This contains two examples of OCP solution with casadi Opti class.
As a general documentation reference for casadi, you can refer to [the official doc](https://web.casadi.org/docs/) and look at the many [examples](https://github.com/casadi/casadi/tree/master/docs/examples)

In particular:
- ```cartpole.py``` uses a custom made problem with analytic dynamics, so it's a standalone example
- ```doublependulm.py``` uses ```pinocchio.casadi``` module (requires ```pinocchio >= 3.0```) to discretize the integrator dynamics with the algorithms implemented in ```pinocchio```

Similarly as shown in the code the functions in ```pinocchio.casadi``` can be used to impose costs or constraints.
