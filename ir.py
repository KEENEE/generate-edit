import ImageReward
import os
import torch

# prompt = "a girl lighting a bonfire on the beach"
prompt = "an elderly couple looking at the sunset"
# prompt = "a traveler observing a blooming cactus in the desert"
# prompt = "a goldenwinged boy flying above a flowerfilled garden"

rootpath = "./images/2/"
image_paths = os.listdir(rootpath)
image_paths = [rootpath + path for path in sorted(image_paths)]

rm = ImageReward.load("ImageReward-v1.0", device=torch.device("cpu"))
rewards = rm.score(prompt, image_paths)
print("Rewards:", rewards)