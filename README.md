# TaskFingerprinting Package For Cognitive Personalization of Task Design

Source code for the  framework built for our study on assessing crowd workers behaviors for task design personalization in crowdsourcing. Beside crowdsourcing, this package can be used to perform cognitive personalization seamlessly in web tasks.
If you use our work in your research, please cite the following paper published at Sensors 2023.

> Dennis Paulino, Diogo Guimarães, António Correia, José Ribeiro, João Barroso and Hugo Paredes. A Model for Cognitive Personalization of Microtask Design. Sensors 2023, 23, 3571. https://doi.org/10.3390/s23073571 

# Example of installing and using the package:

    pip install cognitivepersonalizationtaskfingerprinting
    import cptf.cog_personalization_micro_task_fingerprinting as taskfingerprinting
    import cptf.cog_personalization_deep_learning_model as dl
    
    #It is performed the task fingerprinting technique based on prompt user to enter the path of the interaction log files, and then it will be generated a csv file
    taskfingerprinting.main()
    
    #Based on the task fingerprintings' csv file, the user will be prompted to select the file and develop a deep learning model
    #CAUTION: It is best to look at the source code and adapt the respective parameters to better ajust for each specific dataset
    dl.main()

NOTE: While using the task fingerprinting technique works fine with the published package, for developing the deep learning model it is best to look at the source code available in this repository, in order to adapt the specific parameters of the model.

### Contact

Please feel free to contact Dennis Paulino (dpaulino@utad.pt) for further questions.
