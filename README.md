# CoordConvPPO
This repository implements [Proximal Policy Optimization](https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d) for the **OpenAI CarRacing-v0** environment using an architecture based on [CoordConv layers](https://arxiv.org/abs/1807.03247).

* The **PPO** algorithm is a revision from what was shown in https://github.com/xtma/pytorch_car_caring. I use the same two-headed structure for the output of the network but with a smaller image size and less complexity in the convolutional layers.
* In order to regain the accuracy lost by simplifying the model and the input size I've modified the architecture for using **CoordConv** layers instead of regular convolutional layers. My main reference was this implementation: https://github.com/walsvid/CoordConv

## Performance

The result is that the model is able to **learn much faster** due to a simpler architecture while still **improving the precision** in its behavior for the latest episodes. Due to technical requirements I've set up an experimentation environment of 2000 episodes per run. In this reference the model based on **CoordConv** layers is able to achieve higher accuracy per episode than the baseline:

![compared_models](https://user-images.githubusercontent.com/26325749/145833725-d59ff8c4-2536-4f9e-a6e1-b438737d230c.png)

By taking into account the time of training instead of the episodes you can see that there's a big improvement in computational efficiency. PPO requires a pause in the environment interaction for training the model with some batches of the memory during some epochs (10 epochs in this experiment), which is time-consuming. Training times shown here are measured using a nvidia GEFORCE GTX.


![compared_models_time](https://user-images.githubusercontent.com/26325749/145834890-a18bbedd-aa76-46db-9fa8-b90ef4b15d48.png)

The results shown an improvement on both time and performance. Here you can see an example of the trained model:

![out](https://user-images.githubusercontent.com/26325749/145835032-392da4c0-7a75-4d3c-be32-f18583a1d0ac.gif)

## Usage

The repository is not dockerized since I experienced many issues while executing OpenAI render functions in a container due to its lack of display, as detailed here: https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server. The thing is that CarRacing environment calls the render function inside its step function since the state is the image, and it has to build it, so this call is not avoidable. I've tried many proposed solutions like redirecting the display, using a wrapper or modifying the original OpenAI environment but none of them worked.

In the source folder of the project you'll find the requirements:

```
pip3 install -r requirements.txt
```

Then you can run the training script. It accepts also the --render argument if you want to check the live performance:

```
python3 src/train.py --tag coordconv --img-size 48
```

Once the training is done you can execute the test script. Make sure that you use the correct model name, models are generated and stored in the **models** folder. You don't need to add the extension, only the name of the file. I provide some models in case you don't want to train your own:

```
python3 src/test.py --model coordconv_2021-12-13_11:47:37 --render
```
