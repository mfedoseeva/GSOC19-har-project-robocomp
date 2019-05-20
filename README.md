# Development of Human Activity Recognition Component


## Preliminary Milestones

- [ ] Community bonding: research the solution, get in touch with mentors, sync the goals and brush up the plan, clarify any requirements regarding the tools
- [ ] First iteration of the classification model, training pipeline and robocomp component
- [ ] Improvement of the model, qualitative and quantitative evaluations on the target dataset(s)
- [ ] Finalization and integration of the component, fixing any issues acc. to feedback, testing and documentation


## Review of papers on different architectures and their achieved scores on NTU-RGB-D

 | Name | Link | cs - cv, % acc. | Code | Description | Comment |
 | --- | --- | --- | --- | --- | --- |
 | 2-Stream RNN | https://arxiv.org/pdf/1704.02581.pdf | 71.3 - 79.5 |  https://github.com/hongsong-wang/RNN-for-skeletons | 2 streams with 2 layers of RNN with LSTM, one stream is temporal sequence, where all joints are first grouped in 5 parts (joints within the group are concatenated), then output is concatenated again along the parts to get just the temporal sequence. the second stream takes 'spatial' sequence, which is traversal of the joints starting from the ecntral spine joint. each stream has its own softmax, no fully connected layers to fuse the streams. Use simple data augmentation: rotation, shift, scaling | old lasagne code, but simple idea |
 | Co-occurence learning with CNN | https://arxiv.org/abs/1804.06055 | 86.5 - 91.1 | https://github.com/huguyuehuhu/HCN-pytorch | Also 2-stream, first stream is raw data, the second id joint-wise difference between joint coodrinates at time t+1 and t. Streams are later fused by concatenation along the channels. After two conv layers, the input is transposed so that joints are not channels | ran the reimplementation of this model, got accuracy of 89.3 for cross-view, used smaller batch-size |
 | View adapative NN | https://arxiv.org/abs/1804.07453 | 88.7 - 94.3 | no | skeletons are first processed by a subnetwork that transforms the viewpoint to eliminate the effect of diversity of viewpoints, this is equivalent to transforming the skeleton to a new coordinate system. They make both RNN and CNN versions of the same network and CNN version outperforms RNN. | There is no code and no proper description of the main classification network, so questionable result |
 | Geometric features + LSTMs | https://ieeexplore.ieee.org/document/7926607 | 70 - 82 | no | 8 hand-crafted geometric relation features + 3 layer LSTM | overfitting, not the highest accuracy |
 | Graph convolutional LSTM with attention | https://arxiv.org/abs/1902.09130 | 89 -95 | no | not fully understood yet, skeletons are represented as graphs, inside of LSTM gates are graph convolutional operators  + attention mechanism | need to understand the details yet. no code available |
 | Graph convolutional networks | to do | to do | to do | to do | to do |

 can we try transformer?  



## Datasets

 | Dataset Name | Link | # subjects | # classes | # views | # joints |  # total samples | Comments |
 | --- | --- | --- | --- | --- | --- | --- | --- |
 | SYSU | http://isee.sysu.edu.cn/~hujianfang/ProjectJOULE.html | 40 | 12 | ? | 20 | 480 | most probably single-view. relatively high num of subjects to classes |
 | NTU RGB-D | https://github.com/shahroudy/NTURGB-D, http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp | 40 | 60 | 80 | 25 | 56680 | has to be requested |
 | CAD-60 |  http://pr.cs.cornell.edu/humanactivities/data.php | 4 | 12 | ? | 15 | 60 | most probably single-view. shot in 5 diff. environments |
 | UWA 3D Multiview II | http://staffhome.ecm.uwa.edu.au/~00053650/databases.html | 10 | 30 | 4 | ? | 1076 | most probably 20 joints. multi-view was captured by moving the camera and repeating the cation, rather than filming one action from diff. cameras|
 | MSR Action3D | https://www.uow.edu.au/~wanqing/#Datasets | 10 | 20 | ? | 20 | 567 | most probablu single-view |
 | J-HMDB | http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets | - | 21 | - | 13 | 928 | collected from internet videos, including outdoor environments, but these 928 samples are full body |
 | SBU Kinect | https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html | 7 | 8 | ? | 15 | 300 | entirely **2 person** interaction dataset |
 | PKU-MMD | http://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html | 66 | 51 | 3 | 25 | 1076 | one video is 2-3 mins and contains sequence of different activities |



## Questions for the 1st Meeting

1. what is our use case scenario:
	what is input of the user? rgb video, rgbd video, extracted poses. 
2. are there any special requirements regarding the tools, libraries, environment, etc. 
 

