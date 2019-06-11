## Data separation

CAD-60 dataset divides all data into 5 groups, which they call environments. To each environment only certain number of activities belong. The data itself comes in 4 folders according to the 4 subjects performing the activities.
However the training and testing is performed per environment.  
To seperate the data into separate directories according to the environment group, the data_separation_script can be used.

running the following command will create 5 directories for each environment. inside of each environment there will be 4 directories for each subject. skeleton files will be distributed accordingly. common label map will be saved at the root of the directory where the seperated dataset is stored.

```commandline
python data_separation_script.py
```  

the following lines need to be modified according to your needs in data_seperation_script.py:
```
cad60_path = '../../cad60dataset'
cad60_separated_path = '../../cad60_separated'
```
cad60_path should be the path where the original dataset lies.
cad60_seperated_path should be the directory where the separated dataset should be placed