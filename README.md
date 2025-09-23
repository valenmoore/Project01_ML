<h1>NFL Defensive Analysis</h1>
<i>This project was created using data from the NFL Big Data Bowl 2025. The data has since been removed from Kaggle and it is a violation of license to reupload it to GitHub.</i>

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
  The model was able to succesfully output overall and player-by-player predictions based on individual frames before snap. The model predicted defensive coverage from four options: Cover-1, Cover-2, Cover-3, and Quarters; and offensive formation from seven options: Shotgun, Singleback, Empty, I-Form, Pistol, Jumbo, and Wildcat. Predictions for both offensive and defensive coverage were based only off of defensive players.
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

The primary purpose of the model was zone pass coverage, so less time was spent on the Blitzer and Man-on-Man predictions. This is an area of improvement for the project.

Below is are some video demos of the model. Deep safeties are shown with yellow zones around them; Man defenders are shown with short black arrows; Blitzers are shown with long red arrows. Predicted offensive and defensive formations are shown in the right panel in green if correct.

![ScreenRecording2025-09-23at9 26 31AM-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/1adab242-5609-4c8b-9bd9-ca01e4b0ad25)
![ScreenRecording2025-09-23at10 02 55AM-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/39b9aca1-b097-4eed-a3e2-dd40fef1ba78)

<h2>Installation and Usage</h2>
The project in its entirety is hosted on this Github repository with the exception of the NFL Big Data Bowl Dataset due to license restrictions. I have not found any way to access the dataset online. If you do have access to the dataset, the only change necessary is to alter the paths in the data_compiler.py file. Running any script starting with "train" trains a specified model, as documented in the file. Running main.py will display the results of the model in a Matplotlib window. All Tensorflow models are stored in the /models directory.

<h2>Model Architecture</h2>
The model is built on Tensorflow.
