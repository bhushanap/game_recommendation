
# Game Recommendation System 👾

## Overview 🎮

This project implements a collaborative filtering-based game recommendation system using the fastai library. The recommendation system is trained on a dataset containing user interactions with different games on the Steam gaming platform. The collaborative filtering model learns to predict user preferences based on the playtime hours users have spent on each game.

## Dataset 📊

The dataset used for this project is named steam-200k.csv. It includes information about user interactions with games, such as user IDs, game names, playtime hours, and more. The dataset is preprocessed to filter out non-play interactions and to aggregate playtime hours for unique user-game pairs.

## Data Processing 🛠️

The preprocessing steps involve calculating the total playtime hours for each unique user-game pair and normalizing playtime hours based on the mean playtime for each game. This ensures that the model captures relative preferences rather than absolute playtime.

## Model Architecture 🤖

The recommendation system employs a collaborative filtering model with a bias term. The model is trained using the fastai library's Learner class with the Mean Squared Error (MSE) loss function. The collaborative filtering model uses user and game factors to make predictions.

## Training 🚀

The model is trained for a fixed number of epochs using the fit_one_cycle method.

![Loss](https://github.com/bhushanap/game_recommendation/assets/83635464/f0965c27-4777-4f57-ac38-281679726e38)


The training process aims to minimize the difference between predicted and actual playtime hours. After training, the model factors are visualized using Principal Component Analysis (PCA) to represent games along the most significant feature axes.

![PCA](https://github.com/bhushanap/game_recommendation/assets/83635464/af1af49d-49ed-4c96-ae9c-e0da333b43cf)


## Inference 🎲

After training, the model can be used to make recommendations for games similar to a given input game. The recommendation is based on the cosine similarity between the input game's factors and the factors of other games in the dataset.

## Usage 🕹️

The model is exported and saved as model.pkl. The recommendation system can be used through this Gradio space, allowing users to input a game they like and receive personalized recommendations for similar games.

## Instructions 📝

To train and use the recommendation system:

    1. Run the provided Jupyter Notebook code to train the model.
    2. Export the trained model as model.pkl.
    3. Run the Gradio interface by executing the relevant code in the notebook.
    4. Input a game you like and receive personalized recommendations.

If you want to directly use the model, please visit the huggingface space:

    https://huggingface.co/spaces/bhushanp/game_recommendation?logs=container

## Dependencies 📦

    fastai
    pandas
    numpy
    matplotlib
    torch
    gradio

## Future Plans 🚀

    1. Use a more exhaustive dataset, not just restricted to Steam, for more diverse recommendations.
    2. Fine-tune hyperparameters for better model performance.
    3. Utilize Neural Networks for mapping the embeddings.

Feel free to explore, modify, and contribute to this project! If you have any questions or suggestions, please don't hesitate to reach out. Happy gaming! 🎮✨
