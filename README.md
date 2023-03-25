# Particle Simulation

## Description

The application was written in C++ using the visual studio editor. It uses openGL for rendering and NVIDIA's CUDA API for computation. Because I wanted to focus on the exact solution to the n-body problem, I've decided to implement the O(n^2) algorithm for finding the forces acting on every particle, and because this problem is highly parallelizable, I wrote this simulation to run on the GPU. The simulation allows the user to freely move around and change the parameters in real time.

In future I plan on adding the more efficient space partition algorithm to the simulation. 

## Usage

Using the program is pretty straight forward, as everything is well described and user-friendly. By hovering over an attraction matrix and scrolling up/down, you can edit the forces between pairs of colors. When you change simulation parameters you can apply them by clicking "apply changes".  Below you can see a screenshot of a working simulation.

![app]()

By pressing C, the user can move around the simulation and by pressing ESC the GUI will pop up.
