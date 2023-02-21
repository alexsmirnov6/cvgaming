# Play video games using webcam and CV
### 1. A little bit about idea of the project
The world knows a lot of controllers for playing video games through the movement of the body and its individual parts. But there is no application capable of replacing these controllers with a regular webcam, which almost every PC user has.

Given the interest in such projects from both players and developers, the implementation of this idea seemed very interesting to me. Let's go directly to its analysis.
### 2. Project structure
The entire cycle of implementing the game control mechanism through the camera can be divided into 3 stages:
1) User data collection;
2) Training ML models on the collected data;
3) Assigning keys for the keyboard and mouse emulator, testing the work of trained models;
We will repeat this cycle for two games: the classic ___Mario___ platformer and the ___Punch a Bunch___ boxing simulator. We will control the first game with our palms, and the second with our whole body

### 3. User data collection
First, we will designate the poses that we want to recognize to activate certain commands _(about them in paragraph 5)_. For the Punch a Bunch game initialize 4 poses: __basic stand, block, right and left hook__.
To play Mario, initialize 5 poses. For the right hand: __a state of rest__ and __jump__, for the left hand: __a state of rest, forward and backward movement__

Now let's get down to the data collection itself. Let's go through the list with poses, for one minute we will collect data for each of them and record the results in a .csv file. We will give the user 5 seconds between each pose to change it.

For convenience, I implemented a simple interface: we can see the time remaining to collect data on the current pose, name of current pose and the path along which the data recording.

![output(compress-video-online com)](https://user-images.githubusercontent.com/125807529/220135821-66705092-40b1-4b95-9778-714a27e5d3f9.gif)
![output(compress-video-online com)](https://user-images.githubusercontent.com/125807529/220295600-aae7e99f-7f31-4567-9d6b-4e6a07bef8cd.gif)

### 4. Training models
I use gradient boosting and random forest models built into keras. Linear models and fully-connected neural networks can also be used.

### 5. Using trained models
To interact with the game, you first need to assign a specific set of actions to each of the poses that will be performed in this pose. To do this, I created a dictionary of the format
___pose - request___.
The request consists of nested lists, each of which corresponds to a specific action. One nested query consists of three elements: the device that we are emulating (keyboard or mouse), the action that we want to perform, and the parameter to this action (in our case, the button that we want to hold or release).
___All this can be seen in the script___.

I also wrote a function to decode these requests, consisting almost entirely of if-else constructs (it is also in the script)

Next, using the existing code for getting and outputting labels, we will substitute a trained model, a dictionary of binds and a query decoding function into it. Everything is ready!

Here is an example of the work:

![output(compress-video-online com)](https://user-images.githubusercontent.com/125807529/220127384-440e40bd-32b4-4bc5-a9ca-1d14d0a01ffd.gif)
![output(compress-video-online com)](https://user-images.githubusercontent.com/125807529/220297891-566048b0-6f0a-48cd-88b1-c92822402759.gif)

### 6. Summing up the results

In my opinion, this project has a great potential for development. I plan to continue its development. If it is interesting to you and you want to join, I will be glad of your support
