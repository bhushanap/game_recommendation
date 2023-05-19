import gradio as gr
from fastai.vision.all import *
import pandas as pd
import os
# Load a pre-trained image classification model
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# root = os.path.dirname(__file__)
class DotProductBias(Module):
    def __init__(self, n_users, n_games, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.game_factors = create_params([n_games, n_factors])
        self.game_bias = create_params([n_games])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        games = self.game_factors[x[:,1]]
        res = (users*games).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.game_bias[x[:,1]]
        return res
        return sigmoid_range(res, *self.y_range)


# def get_label(fname):
#     id = int(fname.name[-9:-4])
# #     print(id)
#     cls = int(labels[id-1])-1
# #     print(cls)
#     return name(cls)

learn = load_learner(os.path.join(os.getcwd(),'models','model.pkl'))

merged = pd.read_csv('data.csv')
g = merged.groupby('game')['log_hours'].count()
top_games = g.sort_values(ascending=False).index.values[:100]
top_idxs = tensor([learn.dls.classes['game'].o2i[m] for m in top_games])


# Function to make predictions from an image
# def classify_image(image):
#     # Make a prediction
#     # Decode the prediction and get the class name
#     name = learn.predict(image)
#     return name[0]

# Sample images for user to choose from

def recommend(game):
  if not game in learn.dls.classes['game']:
    output = 'Game not found in Database'
    print(output)
    return output
  tci = learn.dls.classes['game'].o2i[game]
  tco = learn.model.game_factors[tci][None]
  g = merged.groupby('game')['log_hours'].count()
  top_games = g.sort_values(ascending=False).index.values[:100]
  top_idxs = tensor([learn.dls.classes['game'].o2i[m] for m in top_games])
  game_factors = learn.model.game_factors[top_idxs]
  distances = nn.CosineSimilarity(dim=1)(game_factors, tco)
  idx = distances.argsort(descending=True)[0:6].to('cpu')
  # print(learn.dls.classes['game'])
  output = f'If you like {game}, you should try:\
      \n1. {top_games[idx[1]]} - {distances[idx[1]]:.2f}\
      \n2. {top_games[idx[2]]} - {distances[idx[2]]:.2f}\
      \n3. {top_games[idx[3]]} - {distances[idx[3]]:.2f}\
      \n4. {top_games[idx[4]]} - {distances[idx[4]]:.2f}\
      \n5. {top_games[idx[5]]} - {distances[idx[5]]:.2f}'
  print(output)
  return output 


# sample_images = ["./sample_images/AcuraTLType-S2008.jpg", "./sample_images/AudiR8Coupe2012.jpg","./sample_images/DodgeMagnumWagon2008.jpg"]

iface = gr.Interface(
    fn=recommend,
    inputs=gr.Dropdown(list(sorted(top_games)), label="Name of the game you like", info="Will add more games later!"),
    outputs="text",
    live=False,
    title="Game Recommendation System",
    description="Select a game fro the list below",
    examples=['Call of Duty Black Ops', 'Team Fortress 2', 'Dota 2', 'Portal']
)


iface.launch()