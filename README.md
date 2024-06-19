# Monodomain Equation

To run our code, you need to have the following packages installed:
```bash
pip install -r requirements.txt
```

## FEM
To run the FEM code, you need to execute `FEM/FEM.m`. This will run a simulation of the monodomain equation using the Finite Element Method with a mesh of 256, $\Sigma_d = 0.1\Sigma_h$, with 700 timesteps ($dt=0.05$).

Results of the FEM can be found in `FEM/Videos/` for all parameter configurations.

## PINN
To run the PINN code please change directory to `PINN/` and run:
```bash
python train.py
```

A selection of the PINN results can be found in `PINN/good_outputs/`. Please note that rerunning the code will not necessarily produce the same results, as the training is stochastic and the PINN does not always converge to something meaningful.

