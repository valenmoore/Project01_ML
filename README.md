<h1>NFL Defensive Analysis</h1>
<i>This project was created using data from the NFL Big Data Bowl 2025.</i>

<h2>Project Goal</h2>
<p>
  This project uses tracking data from the 2025 NFL Big Data Bowl to analyze defensive coverages from pre-snap behavior. Its main goal is to use player tracking data in real time to predict
  a coverage diagram similar the one showed below:
</p>
<img width="1242" height="659" alt="image" src="https://github.com/user-attachments/assets/16e9f8cf-28cd-47ab-a250-368ddc270a13" />
<p>
  by using predictive models.
</p>
<h2>Model Results</h2>
<p>
  The model was able to succesfully output overall and player-by-player predictions based on individual frames before snap. The model predicted defensive coverage from four options: Cover-1, Cover-2, Cover-3, and Quarters; and offensive formation from seven options: Shotgun, Singleback, Empty, I-Form, Pistol, Jumbo, and Wildcat. These coverages are explained in the "Explaining Football Defensive Coverage" section. Predictions for both offensive and defensive coverage were based only off of defensive players.
  <br>
  The model had the following accuracies for overall team tasks:
</p>

| Task                | Train Accuracy | Val Accuracy |
| ------------------- | -------------- | ------------ |
| Defensive Coverage  | 98.7           | 97.5         |
| Offensive Formation | 98.6           | 96.5         |

Success with offensive formation is particularly significant; the model is able to accurately predict offensive formation just from defensive formation.

The model was mostly successful at predicting binary player-by-player tasks. These tasks included identifying deep safeties, blitzers, and man-on-man defenders.

| Task                | Train Accuracy | Val Accuracy |
| ------------------- | -------------- | ------------ |
| Deep Safeties  | 91.3          | 90.8         |
| Blitzer | 85.2           | 83.6      |
| Man-on-Man | 83.1           | 80.5      |

<h2>Explaining Football Defensive Coverage</h2>
<p>
  The project is built on 
</p>
