# NFL Defensive Analysis

*This project was created using data from the NFL Big Data Bowl 2025. The dataset has since been removed from Kaggle, and it is a violation of the license to re-upload it to GitHub.*

## Project Goal

This project uses tracking data from the 2025 NFL Big Data Bowl to analyze defensive coverages from pre-snap behavior. Its main goal is to use player tracking data in real time to predict a coverage diagram similar to the one shown below:

![Coverage Diagram](https://github.com/user-attachments/assets/16e9f8cf-28cd-47ab-a250-368ddc270a13)

The predictions are generated using machine learning models based on individual player features.

---

## Model Results

The model outputs both overall and player-by-player predictions based on individual frames before snap. It predicts defensive coverage from four options: Cover-1, Cover-2, Cover-3, and Quarters; and offensive formation from seven options: Shotgun, Singleback, Empty, I-Form, Pistol, Jumbo, and Wildcat. Predictions for both tasks were based only on defensive players.

### Team-Level Accuracy

| Task                | Train Accuracy | Val Accuracy |
| ------------------- | -------------- | ------------ |
| Defensive Coverage  | 98.7           | 97.5         |
| Offensive Formation | 98.6           | 96.5         |

Success with offensive formation is particularly significant; the model can accurately predict offensive formation just from defensive positioning.

### Player-Level Accuracy

The model also predicts player-level binary tasks including identifying deep safeties, blitzers, and man-on-man defenders.

| Task          | Train Accuracy | Val Accuracy |
| ------------- | -------------- | ------------ |
| Deep Safeties | 91.3           | 90.8         |
| Blitzer       | 85.2           | 83.6         |
| Man-on-Man    | 83.1           | 80.5         |

Deep safeties are shown with yellow zones, man defenders with short black arrows, and blitzers with long red arrows. Predicted formations are displayed in green if correct.

---

## Model Architecture

The coverage model takes input features for each player (x, y, speed, acceleration, orientation, movement direction, and position encoded as one-hot) and generates a 12-dimensional feature vector for each player. These vectors are flattened for the entire team and passed through a sequence of Dense layers, then classified using a softmax classifier. This architecture was used for both offensive formation and defensive coverage models.

![Coverage Model Architecture](https://github.com/user-attachments/assets/e14bd5d7-5721-4a20-8867-c96df91eb8ba)

The player prediction model uses the same 12-dimensional feature vectors per player, but instead of flattening, each player's features are individually passed through Dense layers followed by a sigmoid classifier. Separate models were trained for Blitzers, Deep Safeties, and Man-on-Man assignments.

![Player Prediction Architecture](https://github.com/user-attachments/assets/c4279159-78ac-4b77-ac52-f5fed1fb0ec4)

Any player with a prediction probability above 75% was classified as a blitzer or man defender. Deep safeties are determined hierarchically based on the predicted coverage.

---

## Installation and Usage

The project is hosted on this GitHub repository, with the exception of the NFL Big Data Bowl Dataset due to license restrictions.  

If you have access to the dataset, update the paths in `data_compiler.py`.  

- Running any script starting with `train` trains a specified model.  
- Running `main.py` displays the model's results in a Matplotlib window.  
- All trained TensorFlow models are stored in the `/models` directory.  
- All files are fully documented; only `main.py` and training scripts can be executed.
