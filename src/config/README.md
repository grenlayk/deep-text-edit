# Configs

In this folder you can find various configuration settings we used in the work process:

- `simple.py` -- just template config because we needed something to be  an example in the beginning 
- `baseline.py` -- this config we used for first runs, it contains implementation of the dummy model with RRDB blocks as generator and only a few loss functions. We used it as a test model to test pipeline and basic loss functions.
- `color.py` -- this is config for colorization model based on UNet architecture. Why do we needed it? -- we used this task to set up work pipeline, write necessary classes and also we used it later to learn how to work with GANs.
- `gan_colorization.py` -- in this config we upgraded colorization model in order to add discriminator and test GAN traing pipeline.
- `typeface.py` -- this config we use to train typeface classificator that we use in typeface loss function.
- `stylegan.py` -- in this config we solved the main task of the project. It containes everything except adversarial loss.