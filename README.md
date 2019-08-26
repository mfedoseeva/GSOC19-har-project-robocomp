# About the Project

This project's goal was to develop a framework for daily human activity recognition as a [Robocomp](https://robocomp.github.io/web) component, which would be based on machine learning approaches achieving comparable score with the state-of-the-art for the selected datasets.

# About the Repository

This repository provides the code for training of the models for Human Activity Recognition Component for Robocomp.  
Code for the components themselves has been done in the Robocomp's forks, these are the pull requests:  
[Interfaces](https://github.com/robocomp/robocomp/pull/224)  
[Components](https://github.com/robocomp/robocomp-robolab/pull/28)  

More specifically, it consists of:
* code for classical machine learning with SVM and hand-crafted features
* code for transfer learning with convolutional neural network
* inference code used in the components (based on both of the approaches)
* summary of relevant research materials

## Commits
[Training related commits](https://github.com/mfedoseeva/GSOC19-har-project-robocomp/commits?author=mfedoseeva)  
[Components related commits 1](https://github.com/robocomp/robocomp/pull/224/commits)  
[Components related commits 2](https://github.com/robocomp/robocomp-robolab/pull/28/commits)

# Use

Both of the training related directories have their own detailed READMEs.  
[Deep learning training](https://github.com/mfedoseeva/GSOC19-har-project-robocomp/blob/master/dl_training/README.md)  
[SVM training](https://github.com/mfedoseeva/GSOC19-har-project-robocomp/blob/master/SVM_hand_crafted/README.md)   
Each of the components also has its own README on installation and use.

# Details

Details about the development of the project are described in the blog posts:  
[Introduction](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post01)  
[First classification attempts](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post02)  
[Improvement of the classification](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post03)  
[Transfer learning and CNN](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post04)  
[Creation of the Components](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post05)

# Results and Future work

Two models have been trained with the maximum accuracy achieved of 94% on the CAD-60 dataset. A group of components described in the [Creation of the Components](https://robocomp.github.io/web/gsoc/2019/mariyam_fedoseeva/post05) has been developed to implement the full pipeline to process and classify data from raw rgb camera input.  
However, despite the good accuracy achieved on the dataset's data, the model's performance is unsatisfactory on real-life input. Future work could focus on making the classifier more robust and general. 



