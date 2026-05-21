Thanks to the GoMLX and GoNB projects we can run a Jupyter notebook that can
execute Go code.

[GoNB](https://github.com/janpfeifer/gonb)  
[GoMLX](https://github.com/gomlx/gomlx)

To run this example, you need to have Go, Jupyter and GoNB installed.

```
$ make install
$ make install-python
```

Once everything is installed you can run the Jupyter notebook:

```
$ make jupyter-run
```

If you are running VS Code, you can use the Jupyter Plugin. Find and install
the plugin and then follow these instructions.

```
Open the `tutorial.ipynb` file

Click on the `Select Kernel` in the top right corner.

Choose Jupyter Kernel

Choose Go(gonb)
```
