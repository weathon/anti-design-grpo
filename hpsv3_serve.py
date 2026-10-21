from flask import Flask, request
from hpsv3 import HPSv3RewardInferencer
inferencer = HPSv3RewardInferencer(device='cuda')

app = Flask(__name__)

@app.route('/')
def home():
    image_paths = [request.args["path"]]
    prompts = [request.args["prompt"]]
    print(image_paths, prompts)
    # Get preference scores
    rewards = inferencer.reward(prompts=prompts, image_paths=image_paths)
    scores = [reward[0].item() for reward in rewards]  # Extract mu values
    return scores

app.run(debug=False)