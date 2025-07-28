
**Graph neural diffusion** </br>
This repo contains the work I've done around [GRAND](https://arxiv.org/abs/2106.10934) by Chamberlain et al. 
[This](diffusion.py) is my initial numpy implementation of the graph diffusion module with both implicit and 
explicit Euler discretization, and
visualization of the process.

After that I [rebuilt](karate.py) it in Pytorch and built the rest of the model, testing it
on the [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) dataset.

The next step is to implement RK4 discretization and try to reproduce the paper's numerics.
